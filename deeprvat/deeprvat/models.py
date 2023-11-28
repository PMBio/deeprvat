import logging
import sys
from collections import OrderedDict
from pprint import pprint
from typing import Any, Dict, List, Optional

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
    def __init__(
        self,
        config: dict,
        n_annotations: Dict[str, int],
        n_covariates: Dict[str, int],
        n_genes: Dict[str, int],
        phenotypes: List[str],
        stage: str = "train",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(config)
        self.save_hyperparameters(kwargs)
        self.save_hyperparameters(
            "n_annotations", "n_covariates", "n_genes", "phenotypes", "stage"
        )

        self.metric_fns = {
            name: METRICS[name]() for name in self.hparams.metrics["all"]
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

    def configure_optimizers(self) -> torch.optim.Optimizer:
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
            lr_scheduler = lr_scheduler_class(
                optimizer, **lrsched_config["config"]
            )

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
        y_pred_by_pheno = self(batch)
        results = dict()
        for name, fn in self.metric_fns.items():
            results[name] = torch.mean(
                torch.stack(
                    [
                        fn(y_pred, batch[pheno]["y"])
                        for pheno, y_pred in y_pred_by_pheno.items()
                    ]
                )
            )
            self.log(f"{self.hparams.stage}_{name}", results[name])

        loss = results[self.hparams.metrics["loss"]]
        if torch.any(torch.isnan(loss)):
            raise RuntimeError("NaNs found in training loss")
        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        y_by_pheno = {
            pheno: pheno_batch["y"] for pheno, pheno_batch in batch.items()
        }
        return {"y_pred_by_pheno": self(batch), "y_by_pheno": y_by_pheno}

    def validation_epoch_end(
        self, prediction_y: List[Dict[str, Dict[str, torch.Tensor]]]
    ):
        y_pred_by_pheno = dict()
        y_by_pheno = dict()
        for result in prediction_y:
            pred = result["y_pred_by_pheno"]
            for pheno, ys in pred.items():
                y_pred_by_pheno[pheno] = torch.cat(
                    [
                        y_pred_by_pheno.get(
                            pheno, torch.tensor([], device=self.device)
                        ),
                        ys,
                    ]
                )

            target = result["y_by_pheno"]
            for pheno, ys in target.items():
                y_by_pheno[pheno] = torch.cat(
                    [
                        y_by_pheno.get(
                            pheno, torch.tensor([], device=self.device)
                        ),
                        ys,
                    ]
                )

        results = dict()
        for name, fn in self.metric_fns.items():
            results[name] = torch.mean(
                torch.stack(
                    [
                        fn(y_pred, y_by_pheno[pheno])
                        for pheno, y_pred in y_pred_by_pheno.items()
                    ]
                )
            )
            self.log(f"val_{name}", results[name])

        self.best_objective = self.objective_operation(
            self.best_objective,
            results[self.hparams.metrics["objective"]].item(),
        )

    def test_step(self, batch: dict, batch_idx: int):
        return {"y_pred": self(batch), "y": batch["y"]}

    def test_epoch_end(self, prediction_y: List[Dict[str, torch.Tensor]]):
        y_pred = torch.cat([p["y_pred"] for p in prediction_y])
        y = torch.cat([p["y"] for p in prediction_y])

        results = {}
        for name, fn in self.metric_fns.items():
            results[name] = fn(y_pred, y)
            self.log(f"val_{name}", results[name])

        self.best_objective = self.objective_operation(
            self.best_objective,
            results[self.hparams.metrics["objective"]].item(),
        )

    def configure_callbacks(self):
        return [ModelSummary()]


class DeepSetAgg(pl.LightningModule):
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
        super().__init__()

        self.output_dim = output_dim
        self.activation = getattr(nn, activation)()
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        self.use_sigmoid = use_sigmoid
        self.reverse = reverse

        input_dim = n_annotations
        phi = []
        for l in range(phi_layers):  # noqa
            output_dim = phi_hidden_dim
            phi.append((f"phi_linear_{l}", nn.Linear(input_dim, output_dim)))
            if dropout is not None:
                phi.append((f"phi_dropout_{l}", self.dropout))
            phi.append((f"phi_activation_{l}", self.activation))
            input_dim = output_dim
        self.phi = nn.Sequential(OrderedDict(phi))

        if pool not in ("sum", "max"):
            raise ValueError(f"Unknown pooling operation {pool}")
        self.pool = pool

        rho = []
        for l in range(rho_layers - 1):  # noqa
            output_dim = rho_hidden_dim
            rho.append((f"rho_linear_{l}", nn.Linear(input_dim, output_dim)))
            if dropout is not None:
                rho.append((f"rho_dropout_{l}", self.dropout))
            rho.append((f"rho_activation_{l}", self.activation))
            input_dim = output_dim
        rho.append(
            (
                f"rho_linear_{rho_layers - 1}",
                nn.Linear(input_dim, self.output_dim),
            )
        )
        self.rho = nn.Sequential(OrderedDict(rho))

    def set_reverse(self, reverse: bool = True):
        self.reverse = reverse

    def forward(self, x):
        x = self.phi(x.permute((0, 1, 3, 2)))
        # x.shape = samples x genes x variants x phi_latent
        if self.pool == "sum":
            x = torch.sum(x, dim=2)
        else:
            x = torch.max(x, dim=2).values
        # Now x.shape = samples x genes x phi_latent
        x = self.rho(x)
        # x.shape = samples x genes x rho_latent
        if self.reverse:
            x = -x
        if self.use_sigmoid:
            x = torch.sigmoid(x)
        return x


class DeepSet(BaseModel):
    def __init__(
        self,
        config: dict,
        n_annotations: Dict[str, int],
        n_covariates: Dict[str, int],
        n_genes: Dict[str, int],
        phenotypes: List[str],
        agg_model: Optional[nn.Module] = None,
        use_sigmoid: bool = False,
        reverse: bool = False,
        **kwargs,
    ):
        super().__init__(
            config, n_annotations, n_covariates, n_genes, phenotypes, **kwargs
        )

        logger.info("Initializing DeepSet model with parameters:")
        pprint(self.hparams)

        activation = get_hparam(self, "activation", "LeakyReLU")
        pool = get_hparam(self, "pool", "sum")
        dropout = get_hparam(self, "dropout", None)

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
        self.agg_model.train(False if self.hparams.stage == "val" else True)

        self.gene_pheno = nn.ModuleDict(
            {
                pheno: nn.Linear(
                    self.hparams.n_covariates + self.hparams.n_genes[pheno], 1
                )
                for pheno in self.hparams.phenotypes
            }
        )

    def forward(self, batch):
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


class DeepSetLinearCommon(BaseModel):
    def __init__(
        self,
        config: dict,
        n_annotations: Dict[str, int],
        n_covariates: Dict[str, int],
        n_genes: Dict[str, int],
        phenotypes: List[str],
        agg_model: Optional[nn.Module] = None,
        use_sigmoid: bool = False,
        reverse: bool = False,
        **kwargs,
    ):
        super().__init__(
            config,
            n_annotations,
            n_covariates + 1,  # adding common vars as one additional covariate
            n_genes,
            phenotypes,
            agg_model,
            use_sigmoid,
            reverse,
            **kwargs,
        )

        # TODO: is common_variants actually part of this_batch?
        self.common_cov = nn.Linear(
            # TODO: figure out correct shape
            this_batch["common_variants"].shape[1], 1
        )

    def forward(self, batch):
        result = dict()
        for pheno, this_batch in batch.items():
            x = this_batch["rare_variant_annotations"]
            # x.shape = samples x genes x annotations x variants
            x = self.agg_model(x).squeeze(dim=2)
            # x.shape = samples x genes
            c = self.common_cov(this_batch["common_varaints"])
            x = torch.cat((this_batch["covariates"], c, x), dim=1)
            # x.shape = samples x (genes + covariates)
            result[pheno] = self.gene_pheno[pheno](x).squeeze(dim=1)
            # result[pheno].shape = samples
        return result


class LinearAgg(pl.LightningModule):
    def __init__(self, n_annotations: int, pool: str, output_dim: int = 1):
        super().__init__()

        self.output_dim = output_dim
        self.pool = pool

        input_dim = n_annotations  # noqa
        self.linear = nn.Linear(n_annotations, self.output_dim)

    def forward(self, x):
        x = self.linear(
            x.permute((0, 1, 3, 2))
        )  # x.shape = samples x genes x variants x output_dim
        if self.pool == "sum":
            x = torch.sum(x, dim=2)
        else:
            x = torch.max(x, dim=2).values
        # Now x.shape = samples x genes x output_dim
        return x


class TwoLayer(BaseModel):
    def __init__(
        self,
        config: dict,
        n_annotations: int,
        n_covariates: int,
        n_genes: int,
        agg_model: Optional[nn.Module] = None,
        **kwargs,
    ):
        super().__init__(
            config, n_annotations, n_covariates, n_genes, **kwargs
        )

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

        self.gene_pheno = nn.Linear(
            self.hparams.n_covariates + self.hparams.n_genes, 1
        )

    def forward(self, batch):
        # samples x genes x annotations x variants
        x = batch["rare_variant_annotations"]
        x = self.agg_model(x).squeeze(dim=2)  # samples x genes
        x = torch.cat((batch["covariates"], x), dim=1)
        x = self.gene_pheno(x).squeeze(dim=1)  # samples
        return x
