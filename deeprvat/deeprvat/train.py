import random
import gc
import logging
import pickle
import sys
from pathlib import Path
from pprint import pformat, pprint
from typing import Dict, Optional, Set, Tuple, Union

import click
import numpy as np
import optuna
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from deeprvat.data import AnnGenoDataModule
from deeprvat.metrics import (
    AveragePrecisionWithLogits,
    PearsonCorr,
    PearsonCorrTorch,
    RSquared,
)
import deeprvat.deeprvat.models_anngeno as deeprvat_models
from deeprvat.utils import suggest_hparams
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn, optim

PathLike = Union[str, Path]

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    level="INFO",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

torch.multiprocessing.set_sharing_strategy("file_system")

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
OPTIMIZERS = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "sparse_adam": optim.SparseAdam,
}
ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
}

DEFAULT_OPTIMIZER = {"type": "adamw", "config": {}}


@click.group()
def cli():
    pass


def train_(
    config: Dict,
    training_regions: Dict[int, np.ndarray],
    log_dir: str,
    sample_set: Optional[Set[str]] = None,
    checkpoint_file: Optional[str] = None,
    trial: Optional[optuna.trial.Trial] = None,
    trial_id: Optional[int] = None,
    debug: bool = False,
    deterministic: bool = False,
) -> Optional[float]:
    """
    Main function called during training. Also used for trial pruning and sampling new parameters in optuna.

    :param config: Dictionary containing configuration parameters, build from YAML file
    :type config: Dict
    :param data: Dict of phenotypes, each containing a dict storing the underlying data.
    :type data: Dict[str, Dict]
    :param log_dir: Path to where logs are written.
    :type log_dir: str
    :param checkpoint_file: Path to where the weights of the trained model should be saved. (optional)
    :type checkpoint_file: Optional[str]
    :param trial: Optuna object generated from the study. (optional)
    :type trial: Optional[optuna.trial.Trial]
    :param trial_id: Current trial in range n_trials. (optional)
    :type trial_id: Optional[int]
    :param debug: Use a strongly reduced dataframe
    :type debug: bool
    :param deterministic: Set random seeds for reproducibility
    :type deterministic: bool

    :returns: Optional[float]: computes the lowest scores of all loss metrics and returns their average
    :rtype: Optional[float]
    """
    anngeno_filename = config["anngeno_file"]

    if deterministic:
        logger.info("Setting random seeds for reproducibility")
        seed = 42
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # if hyperparameter optimization is performed (train(); hpopt_file != None)
    if trial is not None:
        if trial_id is not None:
            # differentiate various repeats in their individual optimization
            trial.set_user_attr("user_id", trial_id)

        # Parameters set in config can be used to indicate hyperparameter optimization.
        # Such cases can be spotted by the following exemplary pattern:
        #
        # phi_hidden_dim: 20
        #       hparam:
        #           type : int
        #               args:
        #                    - 16
        #                    - 64
        #               kwargs:
        #                   step: 16
        #
        # this line should be translated into:
        # phi_layers = optuna.suggest_int(name="phi_hidden_dim", low=16, high=64, step=16)
        # and afterward replace the respective area in config to set the suggestion.
        config["model"]["config"] = suggest_hparams(config["model"]["config"], trial)
        logger.info("Model hyperparameters this trial:")
        pprint(config["model"]["config"])
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        config_out = Path(log_dir) / "model_config.yaml"
        logger.info(f"Writing config to {config_out}")
        with open(config_out, "w") as f:
            yaml.dump(config, f)

    # in practice we only train a single bag, as there are
    # theoretical reasons to omit bagging w.r.t. association testing
    n_bags = config["training"]["n_bags"] if not debug else 3
    train_proportion = config["training"].get("train_proportion", None)
    logger.info(f"Training {n_bags} bagged models")
    results = []
    checkpoint_paths = []
    for k in range(n_bags):
        logger.info(f"  Starting training for bag {k}")

        # load datamodule
        covariates = config["covariates"]
        phenotypes = config["training"]["phenotypes"]
        if isinstance(phenotypes, dict):
            phenotypes = list(phenotypes.keys())
        annotation_columns = config.get("annotations", None)
        dm = AnnGenoDataModule(
            anngeno_filename,
            training_regions=training_regions,
            train_proportion=train_proportion,
            sample_set=sample_set,
            covariates=covariates,
            phenotypes=phenotypes,
            annotation_columns=annotation_columns,
            **config["training"][
                "dataloader_config"
            ],  # batch_size, num_workers, shuffle
        )
        dm.setup(stage="fit")

        # setup the model architecture as specified in config
        model_class = getattr(deeprvat_models, config["model"]["type"])
        model = model_class(
            config=config["model"]["config"],
            n_annotations=len(annotation_columns),
            n_covariates=len(covariates),
            n_genes=sum([rs.shape[0] for rs in training_regions.values()]),
            n_phenotypes=len(phenotypes),
            gene_covariatephenotype_mask=dm.gene_covariatephenotype_mask,
            **config["model"].get("kwargs", {}),
        )

        tb_log_dir = f"{log_dir}/bag_{k}"
        logger.info(f"    Writing TensorBoard logs to {tb_log_dir}")
        tb_logger = TensorBoardLogger(log_dir, name=f"bag_{k}")

        objective = "val_" + config["model"]["config"]["metrics"]["objective"]
        checkpoint_callback = ModelCheckpoint(monitor=objective)
        callbacks = [checkpoint_callback]

        # to prune underperforming trials we enable a pruning strategy that can be set in config
        if "early_stopping" in config["training"]:
            callbacks.append(
                EarlyStopping(monitor=objective, **config["training"]["early_stopping"])
            )

        if debug:
            config["training"]["pl_trainer"]["min_epochs"] = 10
            config["training"]["pl_trainer"]["max_epochs"] = 20

        # initialize trainer, which will call background functionality
        trainer_config = config["training"].get("pl_trainer", {})
        if trainer_config.get("accelerator", None) == "gpu":  # TODO: HACK, fix
            model.gene_pheno.mask = model.gene_pheno.mask.to("cuda:0")
        trainer = pl.Trainer(
            logger=tb_logger,
            callbacks=callbacks,
            **trainer_config,
        )

        while True:
            try:
                import ipdb

                ipdb.set_trace()
                trainer.fit(model, dm)
            except RuntimeError as e:
                # if batch_size is choosen to big, it will be reduced until it fits the GPU
                # TODO: This won't work with code as currently written
                logging.error(f"Caught RuntimeError: {e}")
                if str(e).find("CUDA out of memory") != -1:
                    if dm.hparams.batch_size > 4:
                        logging.error(
                            "Retrying training with half the original batch size"
                        )
                        gc.collect()
                        torch.cuda.empty_cache()
                        dm.hparams.batch_size = dm.hparams.batch_size // 2
                    else:
                        logging.error("Batch size is already <= 4, giving up")
                        raise RuntimeError("Could not find small enough batch size")
                else:
                    logging.error(f"Caught unknown error: {e}")
                    raise e
            else:
                break

        logger.info(
            "Training finished, max GPU memory used: "
            f"{torch.cuda.max_memory_allocated(0)}"
        )

        trial.set_user_attr(
            f"bag_{k}_checkpoint_path", checkpoint_callback.best_model_path
        )
        checkpoint_paths.append(checkpoint_callback.best_model_path)

        if checkpoint_file is not None:
            logger.info(
                f"Symlinking {checkpoint_callback.best_model_path}"
                f" to {checkpoint_file}"
            )
            Path(checkpoint_file).symlink_to(
                Path(checkpoint_callback.best_model_path).resolve()
            )

        results.append(model.best_objective)
        logger.info(f" Result this bag: {model.best_objective}")

        del dm
        gc.collect()
        torch.cuda.empty_cache()

    # Mark checkpoints with worst results to be dropped
    drop_n_bags = config["training"].get("drop_n_bags", None) if not debug else 1
    if drop_n_bags is not None:
        if config["model"]["config"]["metrics"].get("objective_mode", "max") == "max":
            min_result = sorted(results)[drop_n_bags]
            drop_bags = [(r < min_result) for r in results]
        else:
            max_result = sorted(results, reverse=True)[drop_n_bags]
            drop_bags = [(r > max_result) for r in results]

        results = np.array([r for r, d in zip(results, drop_bags) if not d])
        for drop, ckpt in zip(drop_bags, checkpoint_paths):
            if drop:
                Path(ckpt + ".dropped").touch()

    final_result = np.mean(results)
    n_bags_used = n_bags - drop_n_bags if drop_n_bags is not None else n_bags
    logger.info(
        f"Results (top {n_bags_used} bags): "
        f"{final_result} (mean) {np.std(results)} (std)"
    )
    return final_result


@cli.command()
@click.option("--debug", is_flag=True)
@click.option("--deterministic", is_flag=True)
@click.option("--n-trials", type=int, default=1)
@click.option("--trial-id", type=int)
@click.option("--sample-file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--phenotype",
    type=(str, click.Path(exists=True, path_type=Path)),
    multiple=True,
)
@click.argument("config-file", type=click.Path(exists=True, path_type=Path))
@click.argument("log-dir", type=click.Path(path_type=Path))
@click.argument("hpopt-file", type=click.Path(path_type=Path))
def train(
    debug: bool,
    deterministic: bool,
    n_trials: int,
    trial_id: Optional[int],
    sample_file: Optional[str],
    phenotype: Tuple[Tuple[str, Path]],
    config_file: Path,
    log_dir: str,
    hpopt_file: str,
):
    """
    Main function called during training. Also used for trial pruning and sampling new parameters in Optuna.

    :param debug: Use a strongly reduced dataframe
    :type debug: bool
    :param training_gene_file: Path to a pickle file specifying on which genes training should be executed. (optional)
    :type training_gene_file: Optional[str]
    :param n_trials: Number of trials to be performed by the given setting.
    :type n_trials: int
    :param trial_id: Current trial in range n_trials. (optional)
    :type trial_id: Optional[int]
    :param sample_file: Path to a pickle file specifying which samples should be considered during training. (optional)
    :type sample_file: Optional[str]
    :param phenotype: Array of phenotypes, containing an array of paths where the underlying data is stored:
        - str: Phenotype name
        - str: Annotated gene variants as zarr file
        - str: Covariates each sample as zarr file
        - str: Ground truth phenotypes as zarr file
    :type phenotype: Tuple[Tuple[str, str, str, str]]
    :param config_file: Path to a YAML file, which serves for configuration.
    :type config_file: str
    :param log_dir: Path to where logs are stored.
    :type log_dir: str
    :param hpopt_file: Path to where a .db file should be created in which the results of hyperparameter optimization are stored.
    :type hpopt_file: str

    :raises ValueError: If no phenotype option is specified.
    """
    if len(phenotype) == 0:
        raise ValueError("At least one phenotype must be specified")

    with open(config_file) as f:
        config = yaml.safe_load(f)

    if debug:
        config["training"]["pl_trainer"].pop("gpus", None)
        config["training"]["pl_trainer"].pop("precision", None)

    logger.info(f"Running training using config:\n{pformat(config)}")

    logger.info(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory}")

    if sample_file is not None:
        logger.info(f"Using training samples from {sample_file}")
        with open(sample_file, "rb") as f:
            sample_set = set(pickle.load(f)["training_samples"])
        if debug:
            sample_set = set(list(sample_set)[:1000])
    else:
        sample_set = None

    training_regions = {
        pheno: pd.read_parquet(region_file)["id"].to_numpy()
        for pheno, region_file in phenotype
    }

    hparam_optim = config.get("hyperparameter_optimization", None)
    if hparam_optim is None:
        train_(
            config,
            training_regions,
            log_dir,
            sample_set=sample_set,
            debug=debug,
            deterministic=deterministic,
        )
    else:
        pruner_config = config["hyperparameter_optimization"].get("pruning", None)
        if pruner_config is not None:
            pruner: optuna.pruners.BasePruner = getattr(
                optuna.pruners, pruner_config["type"]
            )(**pruner_config["config"])
        else:
            pruner = optuna.pruners.NopPruner()

        objective_direction = config["hyperparameter_optimization"].get(
            "direction", "maximize"
        )

        sampler_config = config["hyperparameter_optimization"].get("sampler", None)
        if sampler_config is not None:
            sampler: optuna.samplers._base.BaseSampler = getattr(
                optuna.samplers, sampler_config["type"]
            )(**sampler_config["config"])
        else:
            sampler = None

        study = optuna.create_study(
            study_name=Path(hpopt_file).stem,
            direction=objective_direction,
            sampler=sampler,
            pruner=pruner,
            storage=f"sqlite:///{hpopt_file}",
            load_if_exists=True,
        )
        study.optimize(
            lambda trial: train_(
                config,
                training_regions,
                log_dir,
                sample_set=sample_set,
                trial=trial,
                trial_id=trial_id,
                debug=debug,
                deterministic=deterministic,
            ),
            n_trials=n_trials,
            timeout=hparam_optim.get("timeout", None),
        )

        logger.info(f"Number of finished trials: {len(study.trials)}")

        trial = study.best_trial
        logger.info(f'Best trial: {trial.user_attrs["user_id"]}')
        logger.info(
            f'  Mean {config["model"]["config"]["metrics"]["objective"]}: '
            f"{trial.value}"
        )
        logger.info(f"  Params:\n{pformat(trial.params)}")


@cli.command()
@click.option("--debug", is_flag=True)
@click.argument("log-dir", type=click.Path())
@click.argument("checkpoint-dir", type=click.Path())
@click.argument("hpopt-db", type=click.Path())
@click.argument("config-file-out", type=click.Path())
def best_training_run(
    debug: bool, log_dir: str, checkpoint_dir: str, hpopt_db: str, config_file_out: str
):
    """
    Function to extract the best trial from an Optuna study and handle associated model checkpoints and configurations.

    :param debug: Use a strongly reduced dataframe
    :type debug: bool
    :param log_dir: Path to where logs are stored.
    :type log_dir: str
    :param checkpoint_dir: Directory where checkpoints have been stored.
    :type checkpoint_dir: str
    :param hpopt_db: Path to the database file containing the Optuna study results.
    :type hpopt_db: str
    :param config_file_out: Path to store a reduced configuration file.
    :type config_file_out: str

    :returns: None
    """

    study = optuna.load_study(
        study_name=Path(hpopt_db).stem, storage=f"sqlite:///{hpopt_db}"
    )

    trials = study.trials_dataframe().query('state == "COMPLETE"')
    with open("deeprvat_config.yaml") as f:
        config = yaml.safe_load(f)
        ascending = (
            False
            if config["hyperparameter_optimization"]["direction"] == "maximize"
            else True
        )
        f.close()
    best_trial = trials.sort_values("value", ascending=ascending).iloc[0]
    best_trial_id = int(best_trial["user_attrs_user_id"])

    logger.info(f"Best trial:\n{best_trial}")

    with open(Path(log_dir) / f"trial{best_trial_id}/model_config.yaml") as f:
        config = yaml.safe_load(f)

    with open(config_file_out, "w") as f:
        yaml.dump({"model": config["model"]}, f)

    n_bags = config["training"]["n_bags"] if not debug else 3
    for k in range(n_bags):
        link_path = Path(checkpoint_dir) / f"bag_{k}.ckpt"
        checkpoint = Path(best_trial[f"user_attrs_bag_{k}_checkpoint_path"])
        link_path.symlink_to(checkpoint.resolve(strict=True))

        # Keep track of models marked to be dropped
        # respective models are not used for downstream processing
        checkpoint_dropped = Path(str(checkpoint) + ".dropped")
        if checkpoint_dropped.is_file():
            dropped_link_path = Path(checkpoint_dir) / f"bag_{k}.ckpt.dropped"
            dropped_link_path.touch()


if __name__ == "__main__":
    cli()
