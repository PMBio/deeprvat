import logging
import numpy as np
import torch.nn.functional as F
import sys
from collections import OrderedDict
from pprint import pprint
from typing import Any, Dict, List, Optional, Self, Union

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary

from deeprvat.metrics import (
    PearsonCorr,
    PearsonCorrTorch,
    RSquared,
    AveragePrecisionWithLogits,
)

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    level="INFO",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

METRICS = {
    "Huber": nn.SmoothL1Loss,
    "MAE": nn.L1Loss,
    "MSE": nn.MSELoss,
    "RSquared": RSquared,
    "PearsonCorr": PearsonCorr,
    "PearsonCorrTorch": PearsonCorrTorch,
    "BCEWithLogits": nn.BCEWithLogitsLoss,
    "AveragePrecisionWithLogits": AveragePrecisionWithLogits,
}


def get_hparam(module: pl.LightningModule, param: str, default: Any):
    if hasattr(module.hparams, param):
        return getattr(module.hparams, param)
    else:
        return default


class BaseModel(pl.LightningModule):
    """
    Base class containing functions that will be called by PyTorch Lightning in the
    background by default.
    """

    def __init__(
        self,
        config: dict,
        n_annotations: int,
        n_covariates: int,
        n_genes: int,
        # n_phenotypes: int,
        training_regions: Optional[Dict[str, np.ndarray]],
        agg_only: bool = False,
        stage: str = "train",
        **kwargs,
    ):
        """
        Initializes BaseModel.

        :param config: Represents the content of config.yaml.
        :type config: dict
        :param n_annotations: Contains the number of annotations used for each phenotype.
        :type n_annotations: Dict[str, int]
        :param n_covariates: Contains the number of covariates used for each phenotype.
        :type n_covariates: Dict[str, int]
        :param n_genes: Contains the number of genes used for each phenotype.
        :type n_genes: Dict[str, int]
        :param phenotypes: Contains the phenotypes used during training.
        :type phenotypes: List[str]
        :param stage: Contains a prefix indicating the dataset the model is operating on. Defaults to "train". (optional)
        :type stage: str
        :param kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.save_hyperparameters(config)
        self.save_hyperparameters(kwargs)
        self.save_hyperparameters(
            "n_annotations",
            "n_covariates",
            "n_genes",
            # "n_phenotypes",
            # "training_regions",
            "agg_only",
            "stage",
        )
        self.training_regions = training_regions

        self.metric_fns = {
            name: METRICS[name](
                reduction="none"
            )  # TODO: Not all metrics support reduction arg
            for name in self.hparams.metrics["all"]
        }

        self.objective_mode = self.hparams.metrics.get("objective_mode", "min")
        if self.objective_mode == "max":
            self.best_objective = float("-inf")
            self.objective_operation = max
        elif self.objective_mode == "min":
            self.best_objective = float("inf")
            self.objective_operation = min
        else:
            raise ValueError("Unknown objective_mode configuration parameter")

        self.validation_step_outputs = []

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Function used to setup an optimizer and scheduler by their
        parameters which are specified in config
        """
        optimizer_config = self.hparams["optimizer"]
        optimizer_class = getattr(torch.optim, optimizer_config["type"])
        optimizer = optimizer_class(
            self.parameters(), **optimizer_config.get("config", {})
        )

        lrsched_config = optimizer_config.get("lr_scheduler", None)
        if lrsched_config is not None:
            lr_scheduler_class = getattr(
                torch.optim.lr_scheduler, lrsched_config["type"]
            )
            lr_scheduler = lr_scheduler_class(optimizer, **lrsched_config["config"])

            if lrsched_config["type"] == "ReduceLROnPlateau":
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": lr_scheduler,
                    "monitor": lrsched_config["monitor"],
                }
            else:
                return [optimizer], [lr_scheduler]
        else:
            return optimizer

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Function called by trainer during training and returns the loss used
        to update weights and biases.

        :param batch: A dictionary containing the batch data.
        :type batch: dict
        :param batch_idx: The index of the current batch.
        :type batch_idx: int

        :returns: torch.Tensor: The loss value computed to update weights and biases
            based on the predictions.
        :raises RuntimeError: If NaNs are found in the training loss.
        """
        import ipdb

        ipdb.set_trace()
        # calls DeepSet.forward()
        y_pred = self(batch)  # n_samples x n_phenos
        results = dict()
        # for all metrics we want to evaluate (specified in config)
        y_true = batch["phenotypes"]
        for name, fn in self.metric_fns.items():
            # compute mean loss across samples and phenotypes
            # ignore loss where true phenotype is NaN
            unreduced_loss = torch.where(y_true.isnan(), 0, fn(y_pred, y_true))
            results[name] = unreduced_loss.sum() / (~y_true.isnan()).sum()
            self.log(f"train_{name}", results[name])
        # set loss from which we compute backward passes
        loss = results[self.hparams.metrics["loss"]]
        if torch.isnan(loss):
            raise RuntimeError("NaNs found in training loss")
        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        """
        During validation, we do not compute backward passes, such that we can accumulate
        phenotype predictions and evaluate them afterward as a whole.

        :param batch: A dictionary containing the validation batch data.
        :type batch: dict
        :param batch_idx: The index of the current validation batch.
        :type batch_idx: int

        :returns: dict: A dictionary containing phenotype predictions ("y_pred_by_pheno")
            and corresponding ground truth values ("y_by_pheno").
        """
        y_pred = self(batch)  # n_samples x n_phenos
        y_true = batch["phenotypes"]
        self.validation_step_outputs.append({"y_pred": y_pred, "y_true": y_true})

    def on_validation_epoch_end(self):
        """
        Evaluate accumulated phenotype predictions at the end of the validation epoch.

        This function takes a list of dictionaries containing accumulated phenotype predictions
        and corresponding ground truth values obtained during the validation process. It computes
        various metrics based on these predictions and logs the results.

        :param prediction_y: A list of dictionaries containing accumulated phenotype predictions
            and corresponding ground truth values obtained during the validation process.
        :type prediction_y: List[Dict[str, Dict[str, torch.Tensor]]]

        :return: None
        :rtype: None
        """
        results = dict()
        y_pred = torch.cat([v["y_pred"] for v in self.validation_step_outputs])
        y_true = torch.cat([v["y_true"] for v in self.validation_step_outputs])
        self.validation_step_outputs.clear()
        for name, fn in self.metric_fns.items():
            # compute mean loss across samples and phenotypes
            # ignore loss where true phenotype is NaN
            unreduced_loss = torch.where(y_true.isnan(), 0, fn(y_pred, y_true))
            results[name] = unreduced_loss.sum() / (~y_true.isnan()).sum()
            self.log(f"val_{name}", results[name])
        # set loss from which we compute backward passes
        loss = results[self.hparams.metrics["loss"]]
        if torch.isnan(loss):
            raise RuntimeError("NaNs found in validation loss")

        # consider all metrics only store the most min/max in self.best_objective
        # to determine if progress was made in the last training epoch.
        self.best_objective = self.objective_operation(
            self.best_objective, results[self.hparams.metrics["objective"]].item()
        )

        # return loss

    def test_step(self, batch: dict, batch_idx: int):
        """
        During testing, we do not compute backward passes, such that we can accumulate
        phenotype predictions and evaluate them afterward as a whole.

        :param batch: A dictionary containing the testing batch data.
        :type batch: dict
        :param batch_idx: The index of the current testing batch.
        :type batch_idx: int

        :returns: dict: A dictionary containing phenotype predictions ("y_pred")
            and corresponding ground truth values ("y").
        :rtype: dict
        """
        return {"y_pred": self(batch), "y": batch["y"]}

    def test_epoch_end(self, prediction_y: List[Dict[str, torch.Tensor]]):
        """
        Evaluate accumulated phenotype predictions at the end of the testing epoch.

        :param prediction_y: A list of dictionaries containing accumulated phenotype predictions
            and corresponding ground truth values obtained during the testing process.
        :type prediction_y: List[Dict[str, Dict[str, torch.Tensor]]]
        """
        y_pred = torch.cat([p["y_pred"] for p in prediction_y])
        y = torch.cat([p["y"] for p in prediction_y])

        results = {}
        for name, fn in self.metric_fns.items():
            results[name] = fn(y_pred, y)
            self.log(f"val_{name}", results[name])

        self.best_objective = self.objective_operation(
            self.best_objective, results[self.hparams.metrics["objective"]].item()
        )

    def configure_callbacks(self):
        return [ModelSummary()]


class DeepSetAgg(pl.LightningModule):
    """
    class contains the gene impairment module used for burden computation.

    Variants are fed through an embedding network Phi to compute a variant embedding.
    The variant embedding is processed by a permutation-invariant aggregation to yield a gene embedding.
    Afterward, the second network Rho estimates the final gene impairment score.
    All parameters of the gene impairment module are shared across genes and traits.
    """

    def __init__(
        self,
        n_annotations: int,
        phi_layers: int,
        phi_hidden_dim: int,
        rho_layers: int,
        rho_hidden_dim: int,
        activation: str,
        pool: str,
        output_dim: int = 1,
        dropout: Optional[float] = None,
        use_sigmoid: bool = False,
        reverse: bool = False,
    ):
        """
        Initializes the DeepSetAgg module.

        :param n_annotations: Number of annotations.
        :type n_annotations: int
        :param phi_layers: Number of layers in Phi.
        :type phi_layers: int
        :param phi_hidden_dim: Internal dimensionality of linear layers in Phi.
        :type phi_hidden_dim: int
        :param rho_layers: Number of layers in Rho.
        :type rho_layers: int
        :param rho_hidden_dim: Internal dimensionality of linear layers in Rho.
        :type rho_hidden_dim: int
        :param activation: Activation function used; should match its name in torch.nn.
        :type activation: str
        :param pool: Invariant aggregation function used to aggregate gene variants. Possible values: 'max', 'sum'.
        :type pool: str
        :param output_dim: Number of burden scores. Defaults to 1. (optional)
        :type output_dim: int
        :param dropout: Probability by which some parameters are set to 0. (optional)
        :type dropout: Optional[float]
        :param use_sigmoid: Whether to project burden scores to [0, 1]. Also used as a linear activation function during training. Defaults to False. (optional)
        :type use_sigmoid: bool
        :param reverse: Whether to reverse the burden score (used during association testing). Defaults to False. (optional)
        :type reverse: bool
        """
        super().__init__()

        self.output_dim = output_dim
        self.activation = getattr(nn, activation)()
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        self.use_sigmoid = use_sigmoid
        self.reverse = reverse

        # setup of Phi
        input_dim = n_annotations
        phi = []
        for l in range(phi_layers):
            output_dim = phi_hidden_dim
            phi.append((f"phi_linear_{l}", nn.Linear(input_dim, output_dim)))
            if dropout is not None:
                phi.append((f"phi_dropout_{l}", self.dropout))
            phi.append((f"phi_activation_{l}", self.activation))
            input_dim = output_dim
        self.phi = nn.Sequential(OrderedDict(phi))

        # setup permutation-invariant aggregation function
        if pool not in ("sum", "max"):
            raise ValueError(f"Unknown pooling operation {pool}")
        self.pool = pool

        # setup of Rho
        rho = []
        for l in range(rho_layers - 1):
            output_dim = rho_hidden_dim
            rho.append((f"rho_linear_{l}", nn.Linear(input_dim, output_dim)))
            if dropout is not None:
                rho.append((f"rho_dropout_{l}", self.dropout))
            rho.append((f"rho_activation_{l}", self.activation))
            input_dim = output_dim
        rho.append(
            (f"rho_linear_{rho_layers - 1}", nn.Linear(input_dim, self.output_dim))
        )
        # No final non-linear activation function to keep the relationship between
        # gene impairment scores and phenotypes linear
        self.rho = nn.Sequential(OrderedDict(rho))

    def set_reverse(self, reverse: bool = True):
        """
        Reverse burden score during association testing if the model predicts in negative space.

        :param reverse: Indicates whether the 'reverse' attribute should be set to True or False.
            Defaults to True.
        :type reverse: bool

        Note:
        Compare associate.py, reverse_models() for further detail
        """
        self.reverse = reverse

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        # genotypes: torch.Tensor,
        # annotations: torch.Tensor,
        # variant_gene_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform a forward pass through the model.

        :param x: Batched input data
        :type x: tensor

        :returns: Burden scores
        :rtype: tensor
        """
        x = self.phi(batch["annotations"])  # variants x phi_latent

        # TODO: test both cases for correctness
        if self.pool == "sum":
            if batch["variant_gene_mask"] is None:
                x = torch.matmul(
                    batch["genotypes"], x
                )  # samples x variants x phi_latent
                x = x.reshape(
                    (x.shape[0], 1, x.shape[1])
                )  # samples x genes x phi_latent
            else:
                x = torch.einsum(
                    "ij,jk->ijk", batch["genotypes"], x
                )  # samples x variants x phi_latent
                # x = torch.einsum(
                #     "ijk,jl->ilk", x, batch["variant_gene_mask"]
                # )  # samples x genes x phi_latent
                x = torch.cat(
                    [
                        torch.sum(z, dim=1, keepdim=True)
                        for z in torch.split(x, batch["region_widths"], dim=1)
                    ],
                    dim=1,
                )
        else:
            raise NotImplementedError  # TODO: Implement!
            # TODO: Maybe make this mask within dataloader
            max_mask = (
                torch.where(batch["genotypes"], 0, float("-inf"))
                .type(x.type())
                .to(x.device)
                .requires_grad_(False)
            )
            # Multiply variant embeddings with genotypes along variant dim
            # Add max_mask - will use broadcasting along embedding (last) dim
            x = torch.max(
                torch.einsum("ijl,lk->ijkl", batch["genotypes"], x) + max_mask, dim=1
            ).values

        x = self.rho(x).squeeze(dim=2)  # samples x genes

        if self.reverse:
            x = -x
        if self.use_sigmoid:
            x = torch.sigmoid(x)

        return x


def forward_single_gene_reference(
    self,
    genotypes: torch.Tensor,  # samples x variants
    annotations: torch.Tensor,  # variants x annotations
) -> torch.Tensor:
    """
    Perform a forward pass through the model - reference implementation,
    works for a single gene only.

    :param x: Batched input data
    :type x: tensor

    :returns: Burden scores
    :rtype: tensor
    """
    variant_embeddings = [
        self.phi(annotations[i]).squeeze() for i in range(annotations.shape[0])
    ]

    gene_embeddings = torch.zeros(genotypes.shape[0], variant_embeddings[0].shape[0])
    for i in range(genotypes.shape[0]):
        for j in range(genotypes.shape[1]):
            gene_embeddings[i] += genotypes[i][j] * variant_embeddings[j]

    x = self.rho(gene_embeddings).squeeze()  # shape (samples, )

    if self.reverse:
        x = -x
    if self.use_sigmoid:
        x = torch.sigmoid(x)

    return x


class DeepSet(BaseModel):
    """
    Wrapper class for burden computation, that also does phenotype prediction.
    It inherits parameters from BaseModel, which is where Pytorch Lightning specific functions
    like "training_step" or "validation_epoch_end" can be found.
    Those functions are called in background by default.
    """

    def __init__(
        self,
        config: dict,
        n_annotations: int,
        n_covariates: int,
        n_genes: int,
        # n_phenotypes: int,
        # gene_covariatephenotype_mask: Optional[torch.Tensor] = None,
        training_regions: Optional[Dict[str, np.ndarray]],
        agg_model: Optional[nn.Module] = None,
        # use_sigmoid: bool = False,
        reverse: bool = False,
        **kwargs,
    ):
        """
        Initialize the DeepSet model.

        :param config: Containing the content of config.yaml.
        :type config: dict
        :param n_annotations: Contains the number of annotations used for each phenotype.
        :type n_annotations: Dict[str, int]
        :param n_covariates: Contains the number of covariates used for each phenotype.
        :type n_covariates: Dict[str, int]
        :param n_genes: Contains the number of genes used for each phenotype.
        :type n_genes: Dict[str, int]
        :param phenotypes: Contains the phenotypes used during training.
        :type phenotypes: List[str]
        :param agg_model: Model used for burden computation. If not provided, it will be initialized. (optional)
        :type agg_model: Optional[pl.LightningModule / nn.Module]
        :param use_sigmoid: Determines if burden scores should be projected to [0, 1]. Acts as a linear activation
            function to mimic association testing during training.
        :type use_sigmoid: bool
        :param reverse: Determines if the burden score should be reversed (used during association testing).
        :type reverse: bool
        :param kwargs: Additional keyword arguments.
        """
        super().__init__(
            config,
            n_annotations,
            n_covariates,
            n_genes,
            training_regions,  # n_phenotypes,
            **kwargs,
        )

        logger.info("Initializing DeepSet model with parameters:")
        pprint(self.hparams)

        activation = get_hparam(self, "activation", "LeakyReLU")
        pool = get_hparam(self, "pool", "sum")
        dropout = get_hparam(self, "dropout", None)
        use_sigmoid = get_hparam(self, "use_sigmoid", False)

        # self.agg_model compresses a batch
        # from: samples x genes x annotations x variants
        # to: samples x genes
        if agg_model is not None:
            self.agg_model = agg_model
        else:
            self.agg_model = DeepSetAgg(
                self.hparams.n_annotations,
                self.hparams.phi_layers,
                self.hparams.phi_hidden_dim,
                self.hparams.rho_layers,
                self.hparams.rho_hidden_dim,
                activation,
                pool,
                dropout=dropout,
                use_sigmoid=use_sigmoid,
                reverse=reverse,
            )
        self.agg_model.train(True if self.hparams.stage == "train" else False)
        # afterwards genes are concatenated with covariates
        # to: samples x (genes + covariates)

        if training_regions is not None:
            self.phenotypes = list(training_regions.keys())
            self.training_region_splits = [
                regions.shape[0] for regions in training_regions.values()
            ]

            # # Use phenotype_mask to only train the weights relevant for each phenotype
            # self.gene_pheno = MaskedLinear(
            #     self.hparams.n_covariates + self.hparams.n_genes,
            #     self.hparams.n_phenotypes,
            #     gene_covariatephenotype_mask,
            # )
            self.gene_pheno = nn.ModuleDict(
                {
                    pheno: nn.Linear(n_covariates + regions.shape[0], 1)
                    for pheno, regions in training_regions.items()
                }
            )

    def forward(self, batch) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the model.

        :param batch: Dictionary of phenotypes, each containing the following keys:
            - indices (tensor): Indices for the underlying dataframe.
            - covariates (tensor): Covariates of samples, e.g., age. Content: samples x covariates.
            - rare_variant_annotations (tensor): Annotated genomic variants. Content: samples x genes x annotations x variants.
            - y (tensor): Actual phenotypes (ground truth data).
        :type batch: dict

        :returns: Dictionary containing predicted phenotypes
        :rtype: dict
        """
        x = self.agg_model(
            batch
            # batch["genotypes"],  # samples x variants
            # batch["annotations"],  # samples x genes
            # batch["variant_gene_mask"],  # variants x genes
        )  # samples x genes

        if not self.hparams.agg_only:
            x_regions = torch.split(x, self.training_region_splits, dim=1)
            result = torch.cat(
                [
                    self.gene_pheno[self.phenotypes[i]](
                        torch.cat((batch["covariates"], x_regions[i]), dim=1)
                    )
                    for i in range(len(self.phenotypes))
                ],
                dim=1,
            )
        else:
            result = x

        return result

    def forward_reference(
        self, batch, training_regions: Dict[str, np.ndarray]
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the model - reference implementation.

        :param batch: Dictionary of phenotypes, each containing the following keys:
            - indices (tensor): Indices for the underlying dataframe.
            - covariates (tensor): Covariates of samples, e.g., age. Content: samples x covariates.
            - rare_variant_annotations (tensor): Annotated genomic variants. Content: samples x genes x annotations x variants.
            - y (tensor): Actual phenotypes (ground truth data).
        :type batch: dict

        :returns: Dictionary containing predicted phenotypes
        :rtype: dict
        """
        assert batch["regions"] == np.concatenate(list(training_regions.values()))
        assert len(batch["regions"]) == len(batch["region_widths"])
        assert len(batch["region_widths"]) == len(training_regions)
        assert [
            x == len(y)
            for x, y in zip(batch["region_widths"], training_regions.values())
        ]

        genotypes_by_gene = torch.split(
            batch["genotypes"], batch["region_widths"], dim=1
        )
        annotations_by_gene = torch.split(
            batch["annotations"], batch["region_widths"], dim=0
        )
        gene_scores = torch.cat(
            [
                self.agg_model.forward_single_gene_reference(g, a)
                for g, a in zip(genotypes_by_gene, annotations_by_gene)
            ]
        )

        if not self.hparams.agg_only:
            # TODO: Split gene_scores, compute for each phenotype

            x_regions = torch.split(x, self.training_region_splits, dim=1)
            result = torch.cat(
                [
                    self.gene_pheno[self.phenotypes[i]](
                        torch.cat((batch["covariates"], x_regions[i]), dim=1)
                    )
                    for i in range(len(self.phenotypes))
                ],
                dim=1,
            )
        else:
            result = x

        return result


class LinearAgg(pl.LightningModule):
    """
    To capture only linear effect, this model can be used as it only uses a single
    linear layer without a non-linear activation function.
    It still contains the gene impairment module used for burden computation.
    """

    def __init__(
        self, n_annotations: int, pool: str, output_dim: int = 1, reverse: bool = False
    ):
        """
        Initialize the LinearAgg model.

        :param n_annotations: Number of annotations.
        :type n_annotations: int
        :param pool: Pooling method ("sum" or "max") to be used.
        :type pool: str
        :param output_dim: Dimensionality of the output. Defaults to 1. (optional)
        :type output_dim: int
        """
        super().__init__()

        self.output_dim = output_dim
        self.pool = pool

        input_dim = n_annotations
        self.linear = nn.Linear(n_annotations, self.output_dim)
        self.reverse = reverse

    def set_reverse(self, reverse: bool = True):
        """
        Reverse burden score during association testing if the model predicts in negative space.

        :param reverse: Indicates whether the 'reverse' attribute should be set to True or False.
            Defaults to True.
        :type reverse: bool

        Note:
        Compare associate.py, reverse_models() for further detail
        """
        self.reverse = reverse

    def forward(self, x):
        """
        Perform a forward pass through the model.

        :param x: Batched input data
        :type x: tensor

        :returns: Burden scores
        :rtype: tensor
        """
        x = self.linear(
            x.permute((0, 1, 3, 2))
        )  # x.shape = samples x genes x variants x output_dim
        if self.pool == "sum":
            x = torch.sum(x, dim=2)
        else:
            x = torch.max(x, dim=2).values
        # Now x.shape = samples x genes x output_dim
        if self.reverse:
            x = -x
        return x


class TwoLayer(BaseModel):
    """
    Wrapper class to capture linear effects. Inherits parameters from BaseModel,
    which is where Pytorch Lightning specific functions like "training_step" or
    "validation_epoch_end" can be found. Those functions are called in background by default.
    """

    def __init__(
        self,
        config: dict,
        n_annotations: int,
        n_covariates: int,
        n_genes: int,
        agg_model: Optional[nn.Module] = None,
        **kwargs,
    ):
        """
        Initializes the TwoLayer model.

        :param config: Represents the content of config.yaml.
        :type config: dict
        :param n_annotations: Number of annotations.
        :type n_annotations: int
        :param n_covariates: Number of covariates.
        :type n_covariates: int
        :param n_genes: Number of genes.
        :type n_genes: int
        :param agg_model: Model used for burden computation. If not provided, it will be initialized. (optional)
        :type agg_model: Optional[nn.Module]
        :param kwargs: Additional keyword arguments.
        """
        super().__init__(config, n_annotations, n_covariates, n_genes, **kwargs)

        logger.info("Initializing TwoLayer model with parameters:")
        pprint(self.hparams)

        n_annotations = self.hparams.n_annotations
        pool = get_hparam(self, "pool", "sum")

        if agg_model is not None:
            self.agg_model = agg_model
        else:
            self.agg_model = LinearAgg(n_annotations, pool)

        if self.hparams.stage == "val":
            self.agg_model.eval()
            for param in self.agg_model.parameters():
                param.requires_grad = False
        else:
            self.agg_model.train()
            for param in self.agg_model.parameters():
                param.requires_grad = True

        self.gene_pheno = nn.ModuleDict(
            {
                pheno: nn.Linear(
                    self.hparams.n_covariates + self.hparams.n_genes[pheno], 1
                )
                for pheno in self.hparams.phenotypes
            }
        )

    def forward(self, batch):
        """
        Forward pass through the model.

        :param batch: Dictionary of phenotypes, each containing the following keys:
            - indices (tensor): Indices for the underlying dataframe.
            - covariates (tensor): Covariates of samples, e.g., age. Content: samples x covariates.
            - rare_variant_annotations (tensor): Annotated genomic variants. Content: samples x genes x annotations x variants.
            - y (tensor): Actual phenotypes (ground truth data).
        :type batch: dict

        :returns: Dictionary containing predicted phenotypes
        :rtype: dict
        """
        result = dict()
        for pheno, this_batch in batch.items():
            x = this_batch["rare_variant_annotations"]
            # x.shape = samples x genes x annotations x variants
            x = self.agg_model(x).squeeze(dim=2)
            # x.shape = samples x genes
            x = torch.cat((this_batch["covariates"], x), dim=1)
            # x.shape = samples x (genes + covariates)
            result[pheno] = self.gene_pheno[pheno](x).squeeze(dim=1)
            # result[pheno].shape = samples
        return result
