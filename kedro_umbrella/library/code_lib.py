from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

import torch
PCA_RATIO = 0.999

def _code_data(data, parameters):
    process = parameters["data_xform"]
    random_state = parameters.get("random_state", None)  # Use provided random state or None
    
    if process == "none":
        # the identity
        model = FunctionTransformer()
    elif process == "std":
        model = StandardScaler(with_mean=False)
    elif process == "pca":
        model = PCA(n_components=PCA_RATIO, random_state = random_state)
    elif process == "pca_std":
        model = make_pipeline(
            PCA(n_components=PCA_RATIO, random_state = random_state), 
            StandardScaler(with_mean=False)
        )
    elif process == "std_pca":
        model = make_pipeline(
            StandardScaler(with_mean=False), 
            PCA(n_components=PCA_RATIO, random_state=random_state)
        )
    return model.fit(data)

class TorchWrapper:
    def __init__(self, xform):
        self.xform = xform

    def transform(self, data):
        return torch.tensor(self.xform.transform(data), dtype=torch.float32)

    def inverse_transform(self, data):
        return torch.tensor(self.xform.inverse_transform(data), dtype=torch.float32)

def xform_data(data, parameters):
    """
    Transforms the given data based on the provided parameters.

    Args:
        data: The input data to be transformed.
        parameters (dict): A dictionary of parameters to control the transformation:

            - to_torch (bool, optional): If True, the transformation will be wrapped
              in a TorchWrapper. Defaults to False.
            - data_xform (str): The type of transformation to apply. Options are:
                - none: No transformation.
                - std: Standard scaling without centering.
                - pca: Principal Component Analysis.
                - pca_std: PCA followed by standard scaling.
                - std_pca: Standard scaling followed by PCA.
            - random_state (int, optional): Random state for reproducibility. Defaults to None.
    Returns:
        tuple: A tuple containing the transform and inverse_transform functions.
    """

    xform = _code_data(data, parameters)
    if parameters.get("to_torch", False):
        wrapper = TorchWrapper(xform)
        return wrapper.transform, wrapper.inverse_transform
    return xform.transform, xform.inverse_transform

def reduce_data(data, parameters):
    # For DesCartesBuilder compatibility, since xform and reduce are different blocks
    # xform for scaling only, reduce if dimensionality reduction
    return xform_data(data, parameters)