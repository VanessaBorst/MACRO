import os
import pathlib
from typing import OrderedDict


# def test_config_path_valid(cli_params):
#     config_file = pathlib.Path(cli_params["--config"])
#     assert config_file.exists()


def test_config_reading_from_args_without_modification(default_config_parser, cli_config_parser_without_modifications):
    # Compare the two ordered dicts which are used when using the default config parser and the one used when using CLI
    # arguments but without modifications
    def walk_dicts(dict1, dict2):
        for i, j in zip(dict1.items(), dict2.items()):
            key_i = i[0]
            value_i = i[1]
            key_j = j[0]
            value_j = j[1]

            assert type(i) == type(j), "Types of the config entries are not matching"
            assert key_i == key_j, "Keys not equal for at least one dict element"
            assert type(value_i) == type(value_j), "Types of the values are not matching for at least one key"

            # If the value is another dict, repeat for the nested dict; otherwise check for equality
            if isinstance(value_i, OrderedDict) and isinstance(value_j, OrderedDict):
                walk_dicts(value_i, value_j)
            else:
                assert value_i == value_j

    walk_dicts(default_config_parser.config, cli_config_parser_without_modifications.config)


def test_config_reading_from_args_with_modification(cli_params, cli_modified_config_parser):
    # check passed args:
    # --config=/home/vanessa/PycharmProjects/2020-ma-vanessa-borst/config.json --device=5 --learning_rate=0.00001
    config_file = pathlib.Path(cli_params["--config"])
    assert config_file.exists()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == cli_params["--device"]
    assert cli_modified_config_parser.config["optimizer"]["args"]["lr"] == float(cli_params["--learning_rate"])

