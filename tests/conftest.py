import argparse
import collections

import pytest
from glob import glob
from utils import read_json
from pathlib import Path

# Important Note 1: No  __init__.py in the test directory!
# Avoid “__init__.py” files in your test directories. This way your tests can run easily against an installed version
# of mypkg, independently from the installed package if it contains the tests or not.
# Run the tests from the root directory with
# "python -m pytest --config=/home/vanessa/PycharmProjects/2020-ma-vanessa-borst/config.json --device=5 --learning_rate=0.00001"
# TODO find out if this is the best way to handle it


# Important Note 2: No fixture imports in conftest.py
# Instead, you can simply create "local pytest plugins" which can be nothing more than Python files with fixtures
from parse_config import ConfigParser


def refactor(string: str) -> str:
    return string.replace("/", ".").replace("\\", ".").replace(".py", "")


pytest_plugins = [
    refactor(fixture) for fixture in glob("tests/fixtures/*.py") if "__" not in fixture
]

_path_to_config = "/home/vanessa/PycharmProjects/2020-ma-vanessa-borst/config.json"


# pytest hook
def pytest_addoption(parser):
    parser.addoption('--config', required=True, type=str,
                     help='config file path (required)')
    parser.addoption('--resume', default=None, type=str,
                     help='path to latest checkpoint (default: None)')
    parser.addoption('--device', default=None, type=str,
                     help='indices of GPUs to enable (default: all)')
    parser.addoption('--learning_rate', default=None, type=str,
                     help='learning rate (default:None)')
    parser.addoption('--batch_size', default=None, type=str,
                     help='batch_size (default: None')


@pytest.fixture
def cli_params(request):
    cli_params = {"--config": request.config.getoption("--config"),
                  "--resume": request.config.getoption("--resume"),
                  "--device": request.config.getoption("--device"),
                  "--learning_rate": request.config.getoption("--learning_rate"),
                  "--batch_size": request.config.getoption("--batch_size")}
    return cli_params


@pytest.fixture(scope='session')
def cli_modified_config_parser():
    arg_parser = argparse.ArgumentParser(description='Read CLI Arguments for Testing')

    arg_parser.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    arg_parser.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    arg_parser.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['-lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['-bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]

    config = ConfigParser.from_args(arg_parser, options, run_id="test_cli_modified_config_parser")
    return config


@pytest.fixture(scope='session')
def cli_config_parser_without_modifications():
    arg_parser = argparse.ArgumentParser(description='CLI Arguments Parser without Modification')
    arg_parser.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    arg_parser.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    arg_parser.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # Do not add custom options here by purpose, as this config parser should not use custom modifications

    config = ConfigParser.from_args(arg_parser, run_id="test_config_parser_without_modifications")
    return config


@pytest.fixture(scope='session')
def default_config_parser():
    return ConfigParser(read_json(_path_to_config), run_id="test_default_config")

