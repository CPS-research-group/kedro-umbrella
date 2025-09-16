import logging
import torch
import numpy as np
import captum
import matplotlib.pyplot as plt
from .utils import ReportDir


logger = logging.getLogger(__name__)

# Variable parameters
NUM_SAMPLE = 1  # number of samples to perform sensitivity analysis
TARGET = 0
DIFF_STEP = 10
GRID_SIZE = 1000
REPORT_DIR = "empty"

class Result:
    def __init__(self, x1=None, x2=None, y1=None, y2=None) -> None:
        self.x1 = x1 if x1 is not None else np.array([])
        self.x2 = x2 if x2 is not None else np.array([])
        self.y1 = y1 if y1 is not None else np.array([])
        self.y2 = y2 if y2 is not None else np.array([])

    def extend(self, res: 'Result') -> None:
        if not isinstance(res, Result):
            raise TypeError("Expected res to be of type Result")
        self.x1 = res.x1 if self.x1.size == 0 else np.vstack((self.x1, res.x1))
        self.x2 = res.x2 if self.x2.size == 0 else np.vstack((self.x2, res.x2))
        self.y1 = res.y1 if self.y1.size == 0 else np.vstack((self.y1, res.y1))
        self.y2 = res.y2 if self.y2.size == 0 else np.vstack((self.y2, res.y2))


    def __str__(self) -> str:
        return (
            f"Result(x1={self.x1}, x2={self.x2}, "
            f"y1={self.y1}, y2={self.y2})"
        )

    def __repr__(self) -> str:
        return self.__str__()

class Model:
    nb = 0

    def __init__(self, regressor: torch.nn.Module, X_inv_xform, Y_inv_xform, params):
        self.model = regressor
        if X_inv_xform and Y_inv_xform:
            self.X_inv_xform = X_inv_xform
            self.Y_inv_xform = Y_inv_xform
        self.NUM_FEATURES = params['NUM_FEATURES']
        self.LOW = np.array(params['LOW'])
        self.HIGH = np.array(params['HIGH'])

    def __call__(self, x):
        return self.model(x)

    def project_back_to_full_space(self, x1, x2, y1, y2):
        """
        Project back to the original point in the physics model
        """
        if (not self.X_inv_xform) or (not self.Y_inv_xform):
            return 

        Model.nb += 1


        logger.info("Projecting back to full space...")
        # Get values in full space
        x1_f = self.X_inv_xform(x1.reshape(1, -1))
        x2_f = self.X_inv_xform(x2.reshape(1, -1))
        y1_f = self.Y_inv_xform(y1.reshape(1, -1))
        y2_f = self.Y_inv_xform(y2.reshape(1, -1))

        from scipy.io import savemat
        savemat(
            f"{REPORT_DIR}/sample_high_sens_{Model.nb}.mat",
            {
                "x1_f": x1_f,
                "x2_f": x2_f,
                "y1_f": y1_f,
                "y2_f": y2_f,
            },
        )

def find_most_sensitive_feature(
    model, low, high, num_features, target=0, method="kernel-shap", num_sample=100
):
    random_np = np.random.uniform(low=low, high=high, size=(num_sample, num_features))

    random_tensor = torch.tensor(random_np).float()

    if method == "ig":
        interp = captum.attr.IntegratedGradients(model)
        attributions = interp.attribute(random_tensor, baselines=0, target=target)
    elif method == "kernel-shap":
        interp = captum.attr.KernelShap(model)
        attributions = interp.attribute(random_tensor, baselines=0, n_samples=100)
    else:
        raise ValueError(f"Unknown method: {method}")
    attributions = attributions.detach().numpy()
    return attributions, random_tensor


def plot_attributions(attributions):
    nb = init_incr_static_cnt(plot_attributions)

    plt.figure()
    plt.imshow(
        attributions.T,
        cmap=plt.cm.seismic,  # pylint: disable=no-member
        interpolation="nearest",
        aspect="auto",
    )
    plt.colorbar(shrink=0.7)
    plt.xlabel("Sample Index")
    plt.ylabel("Feature Index")
    plt.title("Feature Attributions")
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/attribution_{nb}.png")


def aggregate_attributions(attributions, a_abs=True):
    attr = np.abs(attributions) if a_abs else attributions
    attr_by_sample = attr.sum(1)
    top_samples_idx = np.argsort(attr_by_sample)[::-1]
    attr_by_feature = attr.sum(0)
    top_feature_idx = np.argsort(attr_by_feature)[::-1]
    return top_samples_idx, top_feature_idx


def compute_query_range(sample, a_min=None, a_max=None, divide=10):
    """
    Computes the query range for a given sample.
    Args:
        sample (torch.Tensor): The sample to divide the range.
        min (float, optional): The minimum value for clamping. Defaults to None.
        max (float, optional): The maximum value for clamping. Defaults to None.
        divide (int, optional): value to narrow down from sample range.
    Returns:
        tuple: A tuple containing two tensors:
            - query_low (torch.Tensor): The lower bound of the query range, clamped to min.
            - query_high (torch.Tensor): The upper bound of the query range, clamped to max.
    """
    sample_ = sample.detach().numpy() if isinstance(sample, torch.Tensor) else sample

    delta = (a_max - a_min) / divide
    query_min = np.clip(sample_ - delta, a_min=a_min, a_max=a_max)
    query_max = np.clip(sample_ + delta, a_min=a_min, a_max=a_max)
    assert np.all(query_min >= a_min) and np.all(query_max <= a_max)
    return query_min, query_max


def difference_metric(grid, diff_type="classification", step=10):
    """
    Define a difference metric between a point and its neighbors to find
    high elevation area surrounded by flat points
    """

    def _do_diff(grid, diff_type, i, j, neighbors):
        if diff_type == "classification":
            # true for any difference
            diff_ = np.abs(grid[i, j] - neighbors) > 0
        elif diff_type == "regression":
            # keep the numerical value
            diff_ = np.abs(grid[i, j] - neighbors)
        else:
            raise ValueError(f"Unknown diff_type: {diff_type}")
        return diff_

    diff = np.zeros_like(grid)
    diff_pos = np.zeros((grid.shape[0], grid.shape[1], 2), dtype=int)
    for i in range(step, grid.shape[0] - step):
        for j in range(step, grid.shape[1] - step):
            neighbor_idx = np.array(
                [
                    (i - step, j),
                    (i + step, j),
                    (i, j - step),
                    (i, j + step),
                    (i - step, j - step),
                    (i - step, j + step),
                    (i + step, j - step),
                    (i + step, j + step),
                ]
            )
            neighbors = grid[neighbor_idx[:, 0], neighbor_idx[:, 1]]
            diff_ = _do_diff(grid, diff_type, i, j, neighbors)
            max_idx = np.argmax(diff_)
            diff_pos[i, j] = neighbor_idx[max_idx]
            diff[i, j] = diff_[max_idx]

    # Shrink grid to avoid sharp diff at boundary
    return diff[step:-step, step:-step], diff_pos[step:-step, step:-step]


def plot_difference(feat1, feat2, grid, labels, diff_step=10):
    nb = init_incr_static_cnt(plot_difference)

    plt.figure()
    plt.contourf(
        feat1[diff_step:-diff_step], feat2[diff_step:-diff_step], grid, levels=256
    )
    plt.axis("scaled")
    plt.colorbar()
    plt.title("Difference plot")
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/difference_{nb}.png")


def plot_landscape(feat1, feat2, out_reshape, labels):
    nb = init_incr_static_cnt(plot_landscape)

    plt.figure()
    plt.contourf(feat1, feat2, out_reshape, levels=256)
    plt.axis("scaled")
    plt.colorbar()
    plt.title("landscape plot")
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/landscape_{nb}.png")


def plot_policy(grid_0, grid_1, actions, labels):
    nb = init_incr_static_cnt(plot_policy)

    col = ["red", "green", "blue", "orange"]
    colors = [col[i] for i in actions]

    # Plot the scatter plot
    plt.figure()
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plt.scatter(grid_0.flatten(), grid_1.flatten(), c=colors, s=1)

    # Plot properties
    ax.set_title("FCNN Policy")
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.tick_params(axis="both")

    # Get unique colors and create legend handles and labels
    legend_handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=color, markersize=10
        )
        for color in col
    ]
    legend_labels = ["Do Nothing", "Left", "Main", "Right"]

    # Add legend with custom handles and labels
    plt.legend(legend_handles, legend_labels, prop={"weight": "bold", "size": 10})

    # Adjust the layout to make it look clean
    fig.tight_layout()
    plt.savefig(f"{REPORT_DIR}/policy_{nb}.png")


def init_incr_static_cnt(function):
    if not hasattr(function, "nb"):
        function.nb = 0  # Initialize the "static" variable
    function.nb += 1
    nb = function.nb
    return nb

def _get_max_diff_values(dataset, diff_grid, diff_pos, out):
    """
    Get the maximum difference values from the dataset and output arrays.
    Identifies elements with maximum difference and retrieves corresponding 
    values from the dataset and output arrays.
    
    Args:
        dataset (np.ndarray): The original dataset array.
        diff_grid (np.ndarray): The grid of differences.
        diff_pos (np.ndarray): The positions in the original grid corresponding 
                               to the differences.
        out (np.ndarray): The output array corresponding to the dataset.
    
    Returns:
        tuple: A tuple containing the values (x1, x2) from the dataset and 
               (y1, y2) from the output array at the positions of maximum 
               difference.
    """

    # Index w/ max diff on diff space
    diff_flat = diff_grid.flatten()
    diff_sortidx = np.argsort(diff_flat)[::-1]
    idx_max_diff_grid = np.unravel_index(diff_sortidx[0], diff_grid.shape)

    # Map back to the original grid space
    idx1 = tuple(np.array(idx_max_diff_grid) + DIFF_STEP)
    idx2 = tuple(diff_pos[idx_max_diff_grid])

    # Get corresponding input/output to model
    x1, x2 = dataset[idx1], dataset[idx2]
    y1, y2 = out[idx1], out[idx2]

    logger.info(
        f"max diff pos: ({idx_max_diff_grid})\n"
        f"max diff val: {diff_grid[idx_max_diff_grid]}\n"
        f"high diff pair pos: {idx1}, {idx2}\n"
        f"x1 (reduced): {x1}\n"
        f"x2 (reduced): {x2}\n"
        f"y1 (reduced): {y1}\n"
        f"y2 (reduced): {y2}\n"
    )

    return x1, x2, y1, y2

def eval_sensitive_features_grid(model: Model, attr_div, random_div, top_samples_div):
    """
    Step 3: Vary the 2 most-sensitive features in a grid and fix the remaining features

    From focus_around_top_samples, we have top_samples_div, the most important samples
    in a small region of the model. Now, we focus on this region and:
    1. identify important features (feat_imp) for random_div on this sub-domain ("div")
    2. generate a dataset varying the two most important features from their min to mesh.
    We create a mesh of GRID_SIZExGRID_SIZE points for it.

    @in: out from focus_around_top_samples
        - code_X, code_Y: code for displ and eps respectively
        - regressor: the regressor X->Y
        - top_samples_div: top-N important samples in sub-region
        - random_div: the random features in the divided domain
        - attr_div: the attributions in the divided domain
    @out:
        plots for model landscape, difference of feature sensitivity
        and values project back in full space
    """
    res = Result()
    for sample_idx in top_samples_div[:NUM_SAMPLE]:
        logger.info(
            "\n# Step 3: Vary the 2 most-sensitive features in a grid "
            "and fix the remaining features"
        )

        top_sample_div = random_div[sample_idx]
        logger.info(f"random sample {sample_idx}: {top_sample_div}")
        feat_imp = list(np.argsort(attr_div[sample_idx])[::-1])
        logger.info(f"Feature importance for sample: {feat_imp}")

        # Values to vary
        fixed_feat = feat_imp[2:]
        vary_feat = feat_imp[:2]  # the first two most imp.
        feat1 = np.linspace(
            model.LOW[vary_feat[0]], model.HIGH[vary_feat[0]], GRID_SIZE
        )
        feat2 = np.linspace(
            model.LOW[vary_feat[1]], model.HIGH[vary_feat[1]], GRID_SIZE
        )
        grid_0, grid_1 = np.meshgrid(feat1, feat2)

        # Create the dataset w/ fixed feat and varying features
        dataset = np.zeros((GRID_SIZE * GRID_SIZE, model.NUM_FEATURES))
        dataset[:, fixed_feat] = random_div[sample_idx][list(fixed_feat)]
        dataset[:, vary_feat] = np.vstack((grid_0.flatten(), grid_1.flatten())).T
        dataset_tensor = torch.tensor(dataset, dtype=torch.float32)

        # Calculate model output
        out = model.model(dataset_tensor)
        if out.dim() == 1:
            out_reshape = out.detach().numpy().reshape(GRID_SIZE, GRID_SIZE)
        else:
            out_reshape = out[:, TARGET].detach().numpy().reshape(GRID_SIZE, GRID_SIZE)

        # Policy plot
        logger.info("Plotting regression landscape")
        plot_landscape(feat1, feat2, out_reshape, vary_feat)

        # Plot the difference metric
        logger.info("Plotting difference")
        diff_grid, diff_pos = difference_metric(
            out_reshape, diff_type="regression", step=DIFF_STEP
        )
        plot_difference(feat1, feat2, diff_grid, vary_feat, diff_step=DIFF_STEP)

        # TODO watch-out these weird reshapes
        x1, x2, y1, y2 = _get_max_diff_values(
            dataset.reshape((GRID_SIZE, GRID_SIZE, model.NUM_FEATURES)),
            diff_grid, 
            diff_pos, 
            out.detach().numpy().reshape((GRID_SIZE, GRID_SIZE, out.shape[1])))
        res.extend(Result(x1, x2, y1, y2))

        model.project_back_to_full_space(x1, x2, y1, y2)
    return res


def focus_around_top_samples(
    model, attributions, random_tensor, top_samples_idx, method="ig"
):
    """
    Step 2: Focus around each of the top-N important samples.

    Vary around the top-N samples by dividing the radius,
    and determine the top-2 most sensitive features in this sub-region.
    - From the original random_tensor sampling, we focus on the top samples (top_samples_idx)
    - Query the model for sensitive areas in a range of 1/10 of the original range of features

    @in:
        regressor: the model to sample
        attributions: the feature attributions
        random_tensor: random samples
        top_samples_idx: top-N important samples
    @out:
      top_samples_div, top-N important samples in sub-region
      random_div, the random features in the divided domain
      attr_div, the attributions in the divided domain
    """
    all_res = Result()
    for sample_idx in top_samples_idx[:NUM_SAMPLE]:
        logger.info(
            f"\n# Step 2: Focus around each of the top-{NUM_SAMPLE} important samples"
        )
        top_sample = random_tensor[sample_idx]
        logger.info(f"Top important sample {sample_idx} in full domain: {top_sample}")
        feat_imp = np.argsort(attributions[sample_idx])[::-1]
        logger.info(f"Feature importance for sample {sample_idx}: {feat_imp}")

        query_min, query_max = compute_query_range(
            top_sample, a_min=model.LOW, a_max=model.HIGH, divide=10
        )
        logger.info(f"Query range: [{query_min}, {query_max}]")

        # attributions on the divided range
        attr_div, random_div = find_most_sensitive_feature(
            model.model,
            low=query_min,
            high=query_max,
            num_features=model.NUM_FEATURES,
            target=TARGET,
            method=method,
            num_sample=GRID_SIZE,
        )
        plot_attributions(attr_div)

        top_samples_div, top_features_div = aggregate_attributions(attr_div)
        logger.info(
            f"Top-{NUM_SAMPLE} important samples (sub-region): {top_samples_div[:NUM_SAMPLE]}"
        )
        logger.info(f"Most important features (sub-domain): {top_features_div}")

        res = eval_sensitive_features_grid(model, attr_div, random_div, top_samples_div)
        all_res.extend(res)
    return all_res


def calculate_top_samples(model, method="ig"):
    """
    Step 1: Determine the top-N important samples
    @in:
        regressor: the model to sample
    @out:
        random_tensor: random samples used to find most sensitive features
        top_samples_idx: top-N important samples from random_tensor
        attributions: the feature attributions
    """
    logger.info(f"\n# Step 1: Determine the top-{NUM_SAMPLE} important samples")
    attributions, random_tensor = find_most_sensitive_feature(
        model.model,
        low=model.LOW,
        high=model.HIGH,
        num_features=model.NUM_FEATURES,
        target=TARGET,
        method=method,
        num_sample=GRID_SIZE,
    )

    plot_attributions(attributions)

    top_samples_idx, top_feature_idx = aggregate_attributions(attributions)
    logger.info(
        f"Top-{NUM_SAMPLE} important samples (full-domain): {top_samples_idx[:NUM_SAMPLE]}"
    )
    logger.info(f"Most important features (full-domain): {top_feature_idx}")
    return attributions, random_tensor, top_samples_idx


def set_parameters(params):
    global NUM_SAMPLE, TARGET, DIFF_STEP, GRID_SIZE, REPORT_DIR

    NUM_SAMPLE = params.get("num_sample", NUM_SAMPLE)
    TARGET = params.get("target", TARGET)
    DIFF_STEP = params.get("diff_step", DIFF_STEP)
    GRID_SIZE = params.get("grid_size", GRID_SIZE)
    REPORT_DIR = ReportDir(params['_node_name']).get()
    logger.info("# Parameters")
    logger.info(f"NUM_SAMPLE: {NUM_SAMPLE}")
    logger.info(f"TARGET: {TARGET}")
    logger.info(f"DIFF_STEP: {DIFF_STEP}")
    logger.info(f"GRID_SIZE: {GRID_SIZE}")
    logger.info(f"REPORT_DIR: {REPORT_DIR}")

def sensitivity_analysis(model_: torch.nn.Module,  X_inv_xform, Y_inv_xform, 
                         parameters: dict):
    """
    Perform sensitivity analysis on a given model with specified parameters. 
    
    The algorithm aims to find two samples epsilon-close in the input space (x1, x2) that would have significant different in the output space (y1, y2).

    It proceeds as follows: 
        1. Determine top-N important samples using sensitivity analysis method such as Integrated Gradients or KernelShap (from Captum library)
        2. Focus on the most important samples (assume to have high variability) and re-determine sensitivity in a narrow input around each sample
        3. for-each most important sample:
            - find the most sensitivite features
            - evaluate the model response by varying the two most important features and fixing the remaining features 
            - compute the Linf metric (absolute distance between points) and save those w/ the highest value 
    
    Args:
        model_ (torch.nn.Module): The model to be analyzed.
        parameters (dict): A dictionary of parameters to set for the model.
        X_inv_xform, Y_inv_xform: inverse transforms to project back to full space
    
    Returns:
        tuple: A tuple containing the results of the sensitivity analysis:
            - x1: input space sample1
            - x2: input space sample2 (very close to x1).
            - y1: output space sample1 (model output for x1)
            - y2: output space sample2 (model output for x2, most distant from y1)
    
    Typing partition: 
        P1 = {model}
    """

    set_parameters(parameters)

    model = Model(model_, X_inv_xform, Y_inv_xform, parameters)

    # Step 1: Determine the top-N important samples
    attributions, random_tensor, top_samples_idx = calculate_top_samples(model)

    # Step 2: Focus around each of the top-N important samples.
    res = focus_around_top_samples(model, attributions, random_tensor, 
                             top_samples_idx)

    return res.x1, res.x2, res.y1, res.y2

