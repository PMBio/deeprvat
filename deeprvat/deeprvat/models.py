import logging
import sys
from collections import OrderedDict
from pprint import pprint
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary

from deeprvat.utils import init_model
from deeprvat.deeprvat.submodules import Pooling, Layers
from deeprvat.metrics import (
    PearsonCorr,
    PearsonCorrTorch,
    RSquared,
    AveragePrecisionWithLogits,
)

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    level="INFO",
    stream=sys.stdout
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

NORMALIZATION = {
    "spectral_norm": nn.utils.spectral_norm,
    "LayerNorm": nn.LayerNorm
}

def get_hparam(module: pl.LightningModule, param: str, default: Any):
    if hasattr(module.hparams, param):
        return getattr(module.hparams, param)
    else:
        return default

def init_params(hparams, model):
    init_function = getattr(hparams, "init", False)
    if not init_function: return model
    else: return init_model(model, 
                            init_function, 
                            getattr(hparams, "activation", "LeakyReLU"))

class BaseModel(pl.LightningModule):
    def __init__(self,
                 config: dict,
                 n_annotations: Dict[str, int],
                 n_covariates: Dict[str, int],
                 n_genes: Dict[str, int],
                 gene_count: int,
                 max_n_variants: int,
                 phenotypes: List[str],
                 stage: str = "train",
                 **kwargs):
        super().__init__()
        self.save_hyperparameters(config)
        self.save_hyperparameters(kwargs)
        self.save_hyperparameters("n_annotations", "n_covariates", "n_genes",
                                  "gene_count", "max_n_variants",
                                  "phenotypes", "stage")

        self.metric_fns = {name: METRICS[name]()
                           for name in self.hparams.metrics["all"]}

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
        optimizer = optimizer_class(self.parameters(),
                                    **optimizer_config.get("config", {}))

        lrsched_config = optimizer_config.get("lr_scheduler", None)
        if lrsched_config is not None:
            lr_scheduler_class = getattr(torch.optim.lr_scheduler, 
                                         lrsched_config["type"])
            lr_scheduler = lr_scheduler_class(optimizer,
                                              **lrsched_config["config"])

            if lrsched_config["type"] == "ReduceLROnPlateau":
                return {"optimizer": optimizer,
                        "lr_scheduler": lr_scheduler,
                        "monitor": lrsched_config["monitor"]}
            else: return [optimizer], [lr_scheduler]
        else: return optimizer

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        y_pred_by_pheno = self(batch)  
        results = dict()
        for name, fn in self.metric_fns.items():
            results[name] = torch.mean(
                torch.stack([fn(y_pred, batch[pheno]["y"])
                             for pheno, y_pred in y_pred_by_pheno.items()]))
            self.log(f"{self.hparams.stage}_{name}", results[name])

        loss = results[self.hparams.metrics["loss"]]
        if torch.any(torch.isnan(loss)):
            raise RuntimeError("NaNs found in training loss")
        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        y_by_pheno = {pheno: pheno_batch["y"]
                      for pheno, pheno_batch in batch.items()}
        return {"y_pred_by_pheno": self(batch), "y_by_pheno": y_by_pheno}

    def validation_epoch_end(self, prediction_y: List[Dict[str, Dict[str, torch.Tensor]]]):
        y_pred_by_pheno = dict()
        y_by_pheno = dict()
        for result in prediction_y:
            pred = result["y_pred_by_pheno"]
            for pheno, ys in pred.items():
                y_pred_by_pheno[pheno] = torch.cat([
                    y_pred_by_pheno.get(pheno,
                                        torch.tensor([], device=self.device)), ys])

            target = result["y_by_pheno"]
            for pheno, ys in target.items():
                y_by_pheno[pheno] = torch.cat([
                    y_by_pheno.get(pheno, torch.tensor([],device=self.device)), ys])

        results = dict()
        for name, fn in self.metric_fns.items():
            results[name] = torch.mean(
                torch.stack([
                    fn(y_pred, y_by_pheno[pheno])
                    for pheno, y_pred in y_pred_by_pheno.items()]))
            self.log(f"val_{name}", results[name])

        self.best_objective = self.objective_operation(
                self.best_objective, results[self.hparams.metrics["objective"]].item())

    def test_step(self, batch: dict, batch_idx: int):
        return {"y_pred": self(batch), "y": batch["y"]}

    def test_epoch_end(self, prediction_y: List[Dict[str, torch.Tensor]]):
        y_pred = torch.cat([p["y_pred"] for p in prediction_y])
        y = torch.cat([p["y"] for p in prediction_y])

        results = {}
        for name, fn in self.metric_fns.items():
            results[name] = fn(y_pred, y)
            self.log(f"val_{name}", results[name])

        self.best_objective = self.objective_operation(self.best_objective,
            results[self.hparams.metrics["objective"]].item())

    def configure_callbacks(self):
        return [ModelSummary()]

class Phenotype_classifier(pl.LightningModule):
    def __init__(self, hparams, phenotypes, n_genes, gene_count):
        super().__init__()
        # pl.LightningModule already has attribute self.hparams,
        #  which is inherited from its parent class
        self.hparams_ = hparams
        self.phenotypes = phenotypes
        self.n_genes = n_genes
        self.gene_count = gene_count

        self.init_function = getattr(self.hparams_, "init", False)
        self.normalization = getattr(self.hparams_, "normalization", False)
        self.activation = getattr(nn, getattr(self.hparams_, "activation", "LeakyReLU"))()
        self.dropout = getattr(self.hparams_, "classifier_dropout", False)
        if self.dropout: self.drop_layer = nn.Dropout(p=self.dropout)

        self.embed_pheno = hasattr(self.hparams_, "embed_pheno")
        if self.embed_pheno:
            self.pheno2id = dict(zip(phenotypes, range(len(phenotypes))))
            dim = self.hparams_.n_covariates + self.gene_count
            self.burden_pheno_embedding = self.get_embedding(len(phenotypes), dim)
            self.geno_pheno = self.get_model("Classification", dim, 1,
                                             getattr(self.hparams_, "classification_layers", 1), 0)
        else:
            self.geno_pheno = nn.ModuleDict({
                pheno: self.get_model("Classification", self.hparams_.n_covariates + self.hparams_.n_genes[pheno], 1, 
                                      getattr(self.hparams_, "classification_layers", 1), 0)
                for pheno in self.hparams_.phenotypes
            })
    
    def get_embedding(self, in_dim, out_dim):
        embedding = nn.Embedding(in_dim, out_dim)
        return init_params(self.hparams_, embedding)
    
    def pad_genes(self, x, gene_id):
        padding_mask = torch.zeros((x.shape[0], self.gene_count), dtype=x.dtype)
        if x.is_cuda: padding_mask = padding_mask.cuda()
        padding_mask[:, gene_id] = x
        return padding_mask
    
    def get_model(self, prefix, input_dim, output_dim, n_layers, res_layers):
        Layers_obj = Layers(n_layers, res_layers, input_dim, output_dim, self.activation, self.normalization, False, True)
        model = []  
        for l in range(n_layers):         
            model.append((f'{prefix}_layer_{l}', Layers_obj.get_layer(l)))
            if l != n_layers - 1 or prefix != "Classification":
                model.append((f'{prefix}_activation_{l}', self.activation))
        model = nn.Sequential(OrderedDict(model))
        return init_params(self.hparams_, model)

    def forward(self, x, covariates, pheno, gene_id):
        if self.dropout: x = self.drop_layer(x)
        if self.embed_pheno: x = self.pad_genes(x, gene_id)
        x = torch.cat((x, covariates), dim=1)        
        if self.embed_pheno:
            pheno_label = torch.tensor(self.pheno2id[pheno])
            if x.is_cuda: pheno_label = pheno_label.cuda()
            x *= self.burden_pheno_embedding(pheno_label)
            return self.geno_pheno(x).squeeze(dim=1)
        else: return self.geno_pheno[pheno](x).squeeze(dim=1)

class DeepSetAgg(pl.LightningModule):
    def __init__(
        self,
        deep_rvat: int,
        pool_layer: str,
        use_sigmoid: bool = False,
        use_tanh: bool = False,
        reverse: bool = False,
    ):
        super().__init__()

        self.deep_rvat = deep_rvat
        self.pool_layer = pool_layer
        self.use_sigmoid = use_sigmoid
        self.use_tanh = use_tanh
        self.reverse = reverse

    def set_reverse(self, reverse: bool = True):
        self.reverse = reverse

    def forward(self, x):
        x = x.permute((0, 1, 3, 2))
        # x.shape = samples x genes x variants x annotations
        x = self.deep_rvat(x) 
        # x.shape = samples x genes x latent
        if self.reverse: x = -x

        if self.use_sigmoid: x = torch.sigmoid(x)
        if self.use_tanh: x = torch.tanh(x)
        # burden_score
        return x

class DeepSet(BaseModel):
    def __init__(
            self,
            config: dict,
            n_annotations: Dict[str, int],
            n_covariates: Dict[str, int],
            n_genes: Dict[str, int],
            gene_count: int,
            max_n_variants: int,
            phenotypes: List[str],
            agg_model: Optional[nn.Module] = None,
            **kwargs):
        super().__init__(
            config,
            n_annotations,
            n_covariates,
            n_genes,
            gene_count,
            max_n_variants,
            phenotypes,
            **kwargs)
        
        logger.info("Initializing DeepSet model with parameters:")
        pprint(self.hparams)

        self.normalization = getattr(self.hparams, "normalization", False)
        self.activation = getattr(nn, getattr(self.hparams, "activation", "LeakyReLU"))()
        self.use_sigmoid = getattr(self.hparams, "use_sigmoid", False)
        self.use_tanh = getattr(self.hparams, "use_tanh", False)
        self.reverse = getattr(self.hparams, "reverse", False)
        self.pool_layer = getattr(self.hparams, "pool", "sum")
        self.init_power_two = getattr(self.hparams, "first_layer_nearest_power_two", False)
        self.steady_dim = getattr(self.hparams, "steady_dim", False)

        self.phi = self.get_model("phi", 
                                  n_annotations, 
                                  self.hparams.phi_hidden_dim, 
                                  self.hparams.phi_layers, 
                                  self.hparams.phi_res_layers)

        self.pool = Pooling(self.normalization, self.pool_layer, self.hparams.phi_hidden_dim, max_n_variants)
        self.rho = self.get_model("rho", 
                                  self.hparams.phi_hidden_dim, 
                                  self.hparams.rho_hidden_dim, 
                                  self.hparams.rho_layers - 1, 
                                  self.hparams.rho_res_layers)
        self.gene_pheno = Phenotype_classifier(self.hparams, phenotypes, n_genes, gene_count)

        self.deep_rvat = lambda x : self.rho(self.pool(self.phi(x)))

        if agg_model is not None:
            self.agg_model = agg_model
        else:
            self.agg_model = DeepSetAgg(
                deep_rvat=self.deep_rvat,
                pool_layer=self.pool_layer,
                use_sigmoid=self.use_sigmoid,
                use_tanh=self.use_tanh,
                reverse=self.reverse
            )
        self.agg_model.train(False if self.hparams.stage == "val" else True)

        self.train(False if self.hparams.stage == "val" else True)
    
    def get_model(self, prefix, input_dim, output_dim, n_layers, res_layers):
        model = [] 
        Layers_obj = Layers(n_layers, res_layers, input_dim, output_dim, self.activation, self.normalization, self.init_power_two, self.steady_dim)
        for l in range(n_layers):         
            model.append((f"{prefix}_layer_{l}", Layers_obj.get_layer(l)))
            model.append((f"{prefix}_activation_{l}", self.activation))
        if prefix == "rho": model.append((f"{prefix}_linear_{n_layers}", nn.Linear(output_dim, 1)))
        model = nn.Sequential(OrderedDict(model))
        model = init_params(self.hparams, model)
        return model
    
    def forward(self, batch):
        result = dict()
        for pheno, this_batch in batch.items():
            x = this_batch["rare_variant_annotations"] 
            # x.shape = samples x genes x annotations x variants
            burden_score = self.agg_model(x).squeeze(dim=2)
            result[pheno] = self.gene_pheno.forward(burden_score, 
                                                    this_batch["covariates"], 
                                                    pheno, 
                                                    this_batch["gene_id"])
        return result
