import copy
import logging
import os
import math
import shutil
import sys
import pickle
from pprint import pprint
from typing import Any, Callable, Dict, Iterable

import optuna
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import quantile_transform
from statsmodels.stats.multitest import fdrcorrection


logging.basicConfig(
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    level="INFO",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def fdrcorrect_df(group: pd.DataFrame, alpha: float) -> pd.DataFrame:
    group = group.copy()

    rejected, pval_corrected = fdrcorrection(group["pval"], alpha=alpha)
    group["significant"] = rejected
    group["pval_corrected"] = pval_corrected
    return group


def bfcorrect_df(group: pd.DataFrame, alpha: float) -> pd.DataFrame:
    group = group.copy()

    pval_corrected = group["pval"] * len(group)
    group["significant"] = pval_corrected < alpha
    group["pval_corrected"] = pval_corrected
    return group


def pval_correction(group: pd.DataFrame, alpha: float, correction_type: str = "FDR"):
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


def suggest_hparams(config: Dict, trial: optuna.trial.Trial, basename: str = ""):
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
    mean_error = np.mean(errors)
    n = errors.shape[0]
    error_variance = np.mean((errors - mean_error) ** 2) / (n - 1) * n
    return (error_variance / n) ** 0.5


def standardize_series(x: pd.Series) -> pd.Series:
    x = x.astype(np.float32)
    mean = x.mean()
    variance = ((x - mean) ** 2).mean()
    std = variance**0.5
    return (x - mean) / std


def my_quantile_transform(x, seed=1):
    """
    returns Gaussian quantile transformed values, "nan" are kept
    """
    np.random.seed(seed)
    x_transform = x.copy().to_numpy()
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
    x = x.apply(lambda x: (x - mean) / std if x != 0 else 0)
    return x

def calculate_mean_std(x: pd.Series, ignore_zero=True) -> pd.Series:
    x = x.astype(np.float32)
    if ignore_zero:
        x = x[x != float(0)]
    mean = x.mean()
    variance = ((x - mean) ** 2).mean()
    std = variance**0.5
    return std, mean

def normalize_series_with_params(x: pd.Series, min_val, max_val, neg_min_max_calc=False) -> pd.Series:
    if neg_min_max_calc:
        x = x.apply(lambda x: (2 * ((x - min_val) / (max_val - min_val)) ) - 1 if x != 0 else 0)
    else:
        x = x.apply(lambda x: ((x - min_val) / (max_val - min_val)) if x != 0 else 0)
    return x

def calculate_min_max(x: pd.Series) -> pd.Series:
    x = x.astype(np.float32)
    min_val = x.min()
    max_val = x.max()
    return min_val, max_val

def safe_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    validate: str = "1:1",
    equal_row_nums: bool = False,
):
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

def weights_init(init_type='normal', gain=0.02, bias=False, activation="LeakyReLU"):
    r"""Initialize weights in the network.
    Args:
        init_type (str): The name of the initialization scheme.
        gain (float): The parameter that is required for the initialization
            scheme.
        bias (object): If not ``None``, specifies the initialization parameter
            for bias.
    Returns:
        (obj): init function to be applied.
    """
    
    def init_func(m):
        r"""Init function
        Args:
            m: module to be weight initialized.
        """
        class_name = m.__class__.__name__
        if hasattr(m, 'weight') and (
                class_name.find('Conv') != -1 or
                class_name.find('Linear') != -1 or
                class_name.find('Embedding') != -1):
            lr_mul = getattr(m, 'lr_mul', 1.)
            gain_final = gain / lr_mul
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain_final)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain_final)
            elif init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight.data, gain=gain_final)
            elif init_type == 'kaiming':
                if activation == "ReLU":
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                elif activation == "LeakyReLu":
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                # with torch.no_grad(): m.weight.data *= gain_final
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain_final)
            elif init_type == 'none':
                m.reset_parameters()
            else:
                raise NotImplementedError('Initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
            if init_type == 'none': pass
            elif bias is not None:
                bias_type = getattr(bias, 'type', 'normal')
                if bias_type == 'normal':
                    bias_gain = getattr(bias, 'gain', 0.5)
                    nn.init.normal_(m.bias.data, 0.0, bias_gain)
                else:
                    raise NotImplementedError('Initialization method [%s] is not implemented' % bias_type)
            else:
                nn.init.constant_(m.bias.data, 0.0)
    return init_func

def init_model(model, init, activation):
    if init == "none":
        gain = 1
    elif init == "normal":
        gain = 1
    elif init == "xavier":
        gain = math.sqrt(2)
    elif init == "xavier_uniform":
        gain = 1
    elif init  == "orthogonal":
        gain = 1
    elif init  == "kaiming":
        if activation == "ReLU":
            gain =math.sqrt(2) 
        elif activation == "LeakyReLU":
            gain = math.sqrt(1/0.01**2 + 1)
        else:
            raise("Kaiming init has to be used with ReLU or LeakyReLU")
    else:
        pprint("Weight init not implemented.")
        return model
    return model.apply(weights_init(init_type=init, gain=gain, bias=None, activation=activation))

def pad_variants(input_tensor,padding_size):
    #zero pad last dimension to size of max num variants across all phenotypes
    pad_dim = (0, padding_size-input_tensor.shape[-1])
    input_tensor = F.pad(input_tensor,pad_dim,"constant",0)
    return input_tensor

def soft_threshold(l, x):
    return torch.sign(x) * torch.relu(torch.abs(x) - l)

def sign_binary(x):
    ones = torch.ones_like(x)
    return torch.where(x >= 0, ones, -ones)

def prox(v, u, *, lambda_, lambda_bar, M):
    """
    v has shape (m,) or (m, batches)
    u has shape (k,) or (k, batches)

    supports GPU tensors
    """
    onedim = len(v.shape) == 1
    if onedim:
        v = v.unsqueeze(-1)
        u = u.unsqueeze(-1)

    u_abs_sorted = torch.sort(u.abs(), dim=0, descending=True).values

    k, batch = u.shape

    s = torch.arange(k + 1.0).view(-1, 1).to(v)
    zeros = torch.zeros(1, batch).to(u)

    a_s = lambda_ - M * torch.cat(
        [zeros, torch.cumsum(u_abs_sorted - lambda_bar, dim=0)]
    )

    norm_v = torch.norm(v, p=2, dim=0)

    x = F.relu(1 - a_s / norm_v) / (1 + s * M ** 2)

    w = M * x * norm_v
    intervals = soft_threshold(lambda_bar, u_abs_sorted)
    lower = torch.cat([intervals, zeros])

    idx = torch.sum(lower > w, dim=0).unsqueeze(0)

    x_star = torch.gather(x, 0, idx).view(1, batch)
    w_star = torch.gather(w, 0, idx).view(1, batch)

    beta_star = x_star * v
    theta_star = sign_binary(u) * torch.min(soft_threshold(lambda_bar, u.abs()), w_star)

    if onedim:
        beta_star.squeeze_(-1)
        theta_star.squeeze_(-1)

    return beta_star, theta_star

def inplace_prox(beta, theta, lambda_, lambda_bar, M):
    beta.weight.data, theta.weight.data = prox(
        beta.weight.data, theta.weight.data, lambda_=lambda_, lambda_bar=lambda_bar, M=M
    )