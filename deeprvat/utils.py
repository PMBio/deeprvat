import copy
import logging
import os
import math
import shutil
import sys
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Union

import optuna
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import quantile_transform
from statsmodels.stats.multitest import fdrcorrection


logging.basicConfig(
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    level="INFO",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def fdrcorrect_df(group: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """
    Apply False Discovery Rate (FDR) correction to p-values in a DataFrame.

    :param group: DataFrame containing a "pval" column.
    :type group: pd.DataFrame
    :param alpha: Significance level.
    :type alpha: float
    :return: Original DataFrame with additional columns "significant" and "pval_corrected".
    :rtype: pd.DataFrame
    """
    group = group.copy()

    rejected, pval_corrected = fdrcorrection(group["pval"], alpha=alpha)
    group["significant"] = rejected
    group["pval_corrected"] = pval_corrected
    return group


def bfcorrect_df(group: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """
    Apply Bonferroni correction to p-values in a DataFrame.

    :param group: DataFrame containing a "pval" column.
    :type group: pd.DataFrame
    :param alpha: Significance level.
    :type alpha: float
    :return: Original DataFrame with additional columns "significant" and "pval_corrected".
    :rtype: pd.DataFrame
    """
    group = group.copy()

    pval_corrected = group["pval"] * len(group)
    group["significant"] = pval_corrected < alpha
    group["pval_corrected"] = pval_corrected
    return group


def pval_correction(group: pd.DataFrame, alpha: float, correction_type: str = "FDR"):
    """
    Apply p-value correction to a DataFrame.

    :param group: DataFrame containing a column named "pval" with p-values to correct.
    :type group: pd.DataFrame
    :param alpha: Significance level.
    :type alpha: float
    :param correction_type: Type of p-value correction. Options are 'FDR' (default) and 'Bonferroni'.
    :type correction_type: str
    :return: Original DataFrame with additional columns "significant" and "pval_corrected".
    :rtype: pd.DataFrame
    """
    if correction_type == "FDR":
        corrected = fdrcorrect_df(group, alpha)
    elif correction_type == "Bonferroni":
        corrected = bfcorrect_df(group, alpha)
    else:
        raise ValueError(
            f"Unknown correction type: {correction_type}. "
            "Valid values are 'FDR' and 'Bonferroni'."
        )

    corrected["correction_method"] = correction_type
    return corrected


def suggest_hparams(
    config: Dict, trial: optuna.trial.Trial, basename: str = ""
) -> Dict:
    """
    Suggest hyperparameters using Optuna's suggest methods.

    :param config: Configuration dictionary with hyperparameter specifications.
    :type config: Dict
    :param trial: Optuna trial instance.
    :type trial: optuna.trial.Trial
    :param basename: Base name for hyperparameter suggestions.
    :type basename: str
    :return: Updated configuration with suggested hyperparameters.
    :rtype: Dict
    """
    config = copy.deepcopy(config)
    for k, cfg in config.items():
        if isinstance(cfg, dict):
            if list(cfg.keys()) == ["hparam"]:
                suggest_fn = getattr(trial, f'suggest_{cfg["hparam"]["type"]}')
                args_ = cfg["hparam"].get("args", [])
                kwargs_ = cfg["hparam"].get("kwargs", {})
                name = f"{basename}{k}"
                suggestion = suggest_fn(name, *args_, **kwargs_)
                print(f"{name}\t{suggestion}")
                config[k] = suggestion
            else:
                config[k] = suggest_hparams(cfg, trial, f"{basename}{k}/")

    return config


def compute_se(errors: np.ndarray) -> float:
    """
    Compute standard error.

    :param errors: Array of errors.
    :type errors: np.ndarray
    :return: Standard error.
    :rtype: float
    """
    mean_error = np.mean(errors)
    n = errors.shape[0]
    error_variance = np.mean((errors - mean_error) ** 2) / (n - 1) * n
    return (error_variance / n) ** 0.5


def standardize_series(x: pd.Series) -> pd.Series:
    """
    Standardize a pandas Series.

    :param x: Input Series.
    :type x: pd.Series
    :return: Standardized Series.
    :rtype: pd.Series
    """
    x = x.astype(np.float32)
    mean = x.mean()
    variance = ((x - mean) ** 2).mean()
    std = variance**0.5
    return (x - mean) / std


def my_quantile_transform(x, seed=1):
    """
    Gaussian quantile transform for values in a pandas Series.

    :param x: Input pandas Series.
    :type x: pd.Series
    :param seed: Random seed.
    :type seed: int
    :return: Transformed Series.
    :rtype: pd.Series

    .. note::
        "nan" values are kept
    """
    np.random.seed(seed)
    x_transform = x.copy()
    if isinstance(x_transform, pd.Series):
        x_transform = x_transform.to_numpy()

    is_nan = np.isnan(x_transform)
    n_quantiles = np.sum(~is_nan)

    x_transform[~is_nan] = quantile_transform(
        x_transform[~is_nan].reshape([-1, 1]),
        n_quantiles=n_quantiles,
        subsample=n_quantiles,
        output_distribution="normal",
        copy=True,
    )[:, 0]

    return x_transform


def standardize_series_with_params(x: pd.Series, std, mean) -> pd.Series:
    """
    Standardize a pandas Series using provided standard deviation and mean.

    :param x: Input Series.
    :type x: pd.Series
    :param std: Standard deviation to use for standardization.
    :param mean: Mean to use for standardization.
    :return: Standardized Series.
    :rtype: pd.Series
    """
    x = x.apply(lambda x: (x - mean) / std if x != 0 else 0)
    return x


def calculate_mean_std(x: pd.Series, ignore_zero=True) -> pd.Series:
    """
    Calculate mean and standard deviation of a pandas Series.

    :param x: Input Series.
    :type x: pd.Series
    :param ignore_zero: Whether to ignore zero values in calculations, defaults to True.
    :type ignore_zero: bool
    :return: Tuple of standard deviation and mean.
    :rtype: Tuple[float, float]
    """
    x = x.astype(np.float32)
    if ignore_zero:
        x = x[x != float(0)]
    mean = x.mean()
    variance = ((x - mean) ** 2).mean()
    std = variance**0.5
    return std, mean


def safe_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    validate: str = "1:1",
    equal_row_nums: bool = False,
):
    """
    Safely merge two pandas DataFrames.

    :param left: Left DataFrame.
    :type left: pd.DataFrame
    :param right: Right DataFrame.
    :type right: pd.DataFrame
    :param validate: Validation method for the merge.
    :type validate: str
    :param equal_row_nums: Whether to check if the row numbers are equal, defaults to False.
    :type equal_row_nums: bool
    :raises ValueError: If left and right dataframe rows are unequal when 'equal_row_nums' is True.
    :raises RuntimeError: If merged DataFrame has unequal row numbers compared to the left DataFrame.
    :return: Merged DataFrame.
    :rtype: pd.DataFrame
    """
    if equal_row_nums:
        try:
            assert len(left) == len(right)
        except:
            raise ValueError(
                "equal_row_nums is True but row numbers of "
                "left and right dataframes are unequal"
            )

    merged = pd.merge(left, right, validate=validate)

    try:
        assert len(merged) == len(left)
    except:
        raise RuntimeError(
            f"Merged dataframe has {len(merged)} rows, "
            f"left dataframe has {len(left)}"
        )

    return merged


def resolve_path_with_env(path: str) -> str:
    """
    Resolve a path with environment variables.

    :param path: Input path.
    :type path: str
    :return: Resolved path.
    :rtype: str
    """
    path_split = []
    head = path
    while head not in ("", "/"):
        head, tail = os.path.split(head)
        path_split.append(tail)
    if head == "/":
        path_split.append(head)

    path_split = reversed(path_split)
    path_split = [(os.environ[x[1:]] if x.startswith("$") else x) for x in path_split]
    path = os.path.join(*path_split)
    return path


def copy_with_env(path: str, destination: str) -> str:
    """
    Copy a file or directory to a destination with environment variables.

    :param path: Input path (file or directory).
    :type path: str
    :param destination: Destination path.
    :type destination: str
    :return: Resulting destination path.
    :rtype: str
    """
    destination = resolve_path_with_env(destination)

    if os.path.isfile(path):
        basename = os.path.basename(path)
        result = os.path.join(destination, basename)
        if not (
            os.path.exists(result) and os.path.getsize(path) == os.path.getsize(result)
        ):
            shutil.copy2(path, destination)
        return result
    elif os.path.isdir(path):
        _, tail = os.path.split(path)
        result = os.path.join(destination, tail)
        if not (
            os.path.exists(result) and os.path.getsize(path) == os.path.getsize(result)
        ):
            shutil.copytree(path, destination)
        return result
    else:
        raise ValueError("path must be file or dir")


def load_or_init(pickle_file: str, init_fn: Callable) -> Any:
    """
    Load a pickled file or initialize an object.

    :param pickle_file: Pickle file path.
    :type pickle_file: str
    :param init_fn: Initialization function.
    :type init_fn: Callable
    :return: Loaded or initialized object.
    :rtype: Any
    """
    if pickle_file is not None and os.path.isfile(pickle_file):
        logger.info(f"Using pickled file {pickle_file}")
        with open(pickle_file, "rb") as f:
            return pickle.load(f)

    result = init_fn()
    if pickle_file is not None:
        with open(pickle_file, "wb") as f:
            pickle.dump(result, f)
    return result


def remove_prefix(string, prefix):
    """
    Remove a prefix from a string.

    :param string: Input string.
    :type string: str
    :param prefix: Prefix to remove.
    :type prefix: str
    :return: String without the specified prefix.
    :rtype: str
    """
    if string.startswith(prefix):
        return string[len(prefix) :]
    return string


def suggest_batch_size(
    tensor_shape: Iterable[int],
    example: Dict[str, Any] = {
        "batch_size": 16384,
        "tensor_shape": (20, 125, 38),
        "max_mem_bytes": 22_890_098_688,
    },
    buffer_bytes: int = 2_500_000_000,
):
    """
    Suggest a batch size for a tensor based on available GPU memory.

    :param tensor_shape: Shape of the tensor.
    :type tensor_shape: Iterable[int]
    :param example: Example dictionary with batch size, tensor shape, and max memory bytes.
    :type example: Dict[str, Any]
    :param buffer_bytes: Buffer bytes to consider.
    :type buffer_bytes: int
    :return: Suggested batch size for the given tensor shape and GPU memory.
    :rtype: int
    """
    gpu_mem_bytes = torch.cuda.get_device_properties(0).total_memory
    batch_size = math.floor(
        example["batch_size"]
        * ((gpu_mem_bytes - buffer_bytes) / example["max_mem_bytes"])
        * math.prod(example["tensor_shape"])
        / math.prod(tensor_shape)
    )
    logger.info(
        f"Suggested batch size for tensor with shape {tensor_shape} "
        f"and gpu_mem_bytes {gpu_mem_bytes}: {batch_size}"
    )
    return batch_size
