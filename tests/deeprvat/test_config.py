import logging
from pprint import pprint
import pandas as pd
import yaml
import pytest
from click.testing import CliRunner

from pathlib import Path
from deeprvat.deeprvat.config import cli as config_cli
from deeprvat.deeprvat.config import create_main_config, load_yaml

script_dir = Path(__file__).resolve().parent
tests_data_dir = script_dir / "test_data" / "config"


@pytest.mark.parametrize(
    "test_data_name_dir, input_config, clobber",
    [
        (
            "training_only",
            "deeprvat_input_training_config.yaml",
            True,
        ),
        (
            "training_association_testing",
            "deeprvat_input_config.yaml",
            True,
        ),
        (
            "training_association_testing_cv",
            "deeprvat_input_config.yaml",
            True,
        ),
        (
            "association_testing_pretrained_regenie",
            "deeprvat_input_pretrained_models_config.yaml",
            True,
        ),
        (
            "association_testing_pretrained",
            "deeprvat_input_pretrained_models_config.yaml",
            True,
        ),
    ],
)
def test_create_main_config(test_data_name_dir, input_config, clobber, tmp_path):

    current_test_data_dir = tests_data_dir / test_data_name_dir

    config_file_input = current_test_data_dir / "input" / input_config
    expected_config = current_test_data_dir / "expected/deeprvat_config.yaml"

    create_main_config(config_file_input.as_posix(), tmp_path.as_posix(), clobber)

    assert (tmp_path / "deeprvat_config.yaml").exists()

    expected_full_config = load_yaml(expected_config.as_posix())
    generated_config = load_yaml(tmp_path / "deeprvat_config.yaml")
    # nested test on equality
    assert generated_config == expected_full_config
