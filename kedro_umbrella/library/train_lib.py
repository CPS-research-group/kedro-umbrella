from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn import svm
import numpy as np
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import logging


logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)  # Skip INFO logs.
optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

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


def _objective(trial, X, Y, parameters):
    model = parameters["model"]
    if model == "mlp2":
        parameters["hidden_layer_sizes"] = tuple(
            trial.suggest_int(f"n_units_l{i}", 10, 200) for i in range(1, 4)
        )
        parameters["learning_rate_init"] = trial.suggest_float(
            "learning_rate_init", 0.0001, 0.1
        )
    elif model == "mlp1":
        parameters["hidden_layer_sizes"] = tuple(
            trial.suggest_int(f"n_units_l{i}", 50, 150) for i in range(1, 4)
        )
    elif model == "mlp":
        parameters["hidden_layer_sizes"] = tuple(
            trial.suggest_int(f"n_units_l{i}", 10, 100) for i in range(1, 3)
        )
    elif model == "dt":
        parameters["max_depth"] = trial.suggest_int("max_depth", 1, 20)
        parameters["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 10)
    elif model == "svr":
        parameters["C"] = trial.suggest_float("C", 0.1, 10.0)
        parameters["epsilon"] = trial.suggest_float("epsilon", 0.01, 1.0)

    X_train, X_valid, Y_train, Y_valid = train_test_split(
        X, Y, test_size=0.2, random_state=parameters["random_state"]
    )
    model = _train_model(X_train, Y_train, parameters)
    Y_pred = model.predict(X_valid)
    return mean_squared_error(Y_valid, Y_pred)


def basic_trainer(X, Y, parameters):
    """
    Train a model with given data and parameters, optionally using Optuna for hyperparameter optimization. This wraps the Scikit-Learn implementation for access in the Builder. 

    Args:
        X (pd.DataFrame or np.ndarray): Feature matrix.
        Y (pd.Series or np.ndarray): Target vector.
        parameters (dict): Dictionary containing model parameters and training options:

            - "use_optuna" (bool, optional): Whether to use Optuna for hyperparameter optimization. Default is False.
            - "n_trials" (int, optional): Number of trials for Optuna optimization. Default is 50.
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

    use_optuna = parameters.get("use_optuna", False)
    n_trials = parameters.get("n_trials", 50)

    if not use_optuna:
        return _train_model(X, Y, parameters).predict

    model_type = parameters["model"]
    study = optuna.create_study(direction="minimize")

    logger.info(f"Optimizing hyperparameters for {model_type} model")

    study.optimize(
        lambda trial: _objective(trial, X, Y, parameters),
        n_trials=n_trials,
        # show_progress_bar=True,
    )
    best_params = study.best_trial.params

    # Update parameters with the best hyperparameters found by Optuna
    if model_type in ["mlp2", "mlp", "mlp1"]:
        parameters["hidden_layer_sizes"] = tuple(
            best_params[f"n_units_l{i}"]
            for i in range(1, 4)
            if f"n_units_l{i}" in best_params
        )
    if model_type == "mlp2":
        parameters["learning_rate_init"] = best_params["learning_rate_init"]
    elif model_type == "dt":
        parameters["max_depth"] = best_params["max_depth"]
        parameters["min_samples_split"] = best_params["min_samples_split"]
    elif model_type == "svr":
        parameters["C"] = best_params["C"]
        parameters["epsilon"] = best_params["epsilon"]

    return _train_model(X, Y, parameters).predict
