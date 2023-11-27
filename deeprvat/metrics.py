import logging
import sys

import torch
import torch.nn.functional as F
from scipy.stats.stats import pearsonr
from sklearn.metrics import average_precision_score

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    level="INFO",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


class RSquared:
    """
    Calculates the R-squared (coefficient of determination) between predictions and targets.
    """
    def __init__(self):
        pass

    def __call__(self, preds: torch.tensor, targets: torch.tensor):
        """
        Calculate R-squared value between two tensors.

        Parameters:
        - preds (torch.tensor): Tensor containing predicted values.
        - targets (torch.tensor): Tensor containing target values.

        Returns:
        torch.tensor: R-squared value.
        """
        y_mean = torch.mean(targets)
        ss_tot = torch.sum(torch.square(targets - y_mean))
        ss_res = torch.sum(torch.square(targets - preds))
        return 1 - ss_res / ss_tot


class PearsonCorr:
    """
    Calculates the Pearson correlation coefficient between burdens and targets.
    """
    def __init__(self):
        pass

    def __call__(self, burden, y):
        """
        Calculate Pearson correlation coefficient.

        Parameters:
        - burden (torch.tensor): Tensor containing burden values.
        - y (torch.tensor): Tensor containing target values.

        Returns:
        float: Pearson correlation coefficient.
        """
        if len(burden.shape) > 1:  # was the burden computed for >1 genes
            corrs = []
            for i in range(burden.shape[1]):  # number of genes
                b = burden[:, i].squeeze()
                if len(b.unique()) <= 1:
                    corr = 0
                else:
                    corr = abs(pearsonr(b, y.squeeze())[0])
                corrs.append(corr)
            corr = sum(corrs)
            logger.info(f"correlation_sum: {corr}")
        else:
            corr = abs(pearsonr(burden.squeeze(), y.squeeze())[0])

        return corr


class PearsonCorrTorch:
    """
    Calculates the Pearson correlation coefficient between burdens and targets using PyTorch tensor operations.
    """
    def __init__(self):
        pass

    def __call__(self, burden, y):
        """
        Calculate Pearson correlation coefficient using PyTorch tensor operations.

        Parameters:
        - burden (torch.tensor): Tensor containing burden values.
        - y (torch.tensor): Tensor containing target values.

        Returns:
        torch.tensor: Pearson correlation coefficient.
        """
        if len(burden.shape) > 1:  # was the burden computed for >1 genes
            corrs = []
            for i in range(burden.shape[1]):  # number of genes
                b = burden[:, i].squeeze()
                if (
                    len(b.unique()) <= 1
                ):  # if all burden values are the same, correlation will be nan -> must be avoided
                    corr = torch.tensor(0)
                else:
                    corr = abs(self.calculate_pearsonr(b, y.squeeze()))
                corrs.append(corr)
            # corr = sum(corrs)
            corr = torch.stack(corrs).mean()
            logger.info(f"correlation_sum: {corr}")

        else:
            corr = abs(self.calculate_pearsonr(burden.squeeze(), y.squeeze()))

        return corr

    def calculate_pearsonr(self, x, y):
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        corr = torch.sum(vx * vy) / (
            torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2))
        )
        return corr


class AveragePrecisionWithLogits:
    """
    Calculates the average precision score between logits and targets.
    """
    def __init__(self):
        pass

    def __call__(self, logits, y):
        """
        Calculate average precision score.

        Parameters:
        - logits (torch.tensor): Tensor containing logits.
        - y (torch.tensor): Tensor containing target values.

        Returns:
        float: Average precision score.
        """
        y_scores = F.sigmoid(logits.detach())
        return average_precision_score(y.detach().cpu().numpy(), y_scores.cpu().numpy())
