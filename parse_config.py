import logging
import os
from datetime import datetime
from functools import reduce, partial
from operator import getitem
from pathlib import Path

from logger import setup_logging
from utils import get_project_root, read_json, write_json


class ConfigParser:
    def __init__(self, config, resume=None, modification=None, mode=None, use_tune=False, run_id=None,
                 create_save_log_dir = True):
        """
        class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.
        :param config: Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.
        :param resume: String, path to the checkpoint being loaded.
        :param modification: Dict keychain:value, specifying position values to be replaced from config dict.
        :param run_id: Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default
        """
        # load config file and apply modification
        self._config = _update_config(config, modification)
        self._resume = resume
        self._use_tune = use_tune

        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config['trainer']['save_dir'])

        exper_name = self.config['name']
        if run_id is None:  # use timestamp as default run-id
            run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        details = "_{}_bs{}{}".format("ml" if self.config['arch']['args']['multi_label_training'] else "sl",
                                      self.config['data_loader']['args']['batch_size'],
                                      self.config['run_details'])

        if create_save_log_dir:
            # Training
            if mode is None or mode=='train':
                self._save_dir = Path(
                    os.path.join(get_project_root(), save_dir / 'models' / exper_name / str(run_id + details)))
                self._log_dir = Path(
                    os.path.join(get_project_root(), save_dir / 'log' / exper_name / str(run_id + details)))
            else:
                self._save_dir=None
                self._log_dir=None

            # Testing
            if mode is not None and mode=='test':
                assert resume is not None, "checkpoint must be provided for testing"
                assert 'valid' in self.config['data_loader']['test_dir'].lower() or \
                       'test' in self.config['data_loader']['test_dir'].lower(), "Path should link validation or test dir"
                self._test_output_dir = Path(os.path.join(resume.parent, 'test_output')) if \
                    'test' in self.config['data_loader']['test_dir'].lower()                 \
                    else Path(os.path.join(resume.parent, 'valid_output'))
            else:
                self._test_output_dir = None    # For training not needed

            # make directory for saving checkpoints and log and test outputs (if needed).
            exist_ok = run_id == ''
            if self._save_dir is not None:
                self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
            if self._log_dir is not None:
                self.log_dir.mkdir(parents=True, exist_ok=exist_ok)
            if self.test_output_dir is not None:
                self.test_output_dir.mkdir(parents=True, exist_ok=True)

        else:
            self._save_dir = None
            self._log_dir = None
            self._test_output_dir = None

        # save updated config file to the checkpoint dir
        if self._save_dir is not None:
            write_json(self.config, self.save_dir / 'config.json')

        # if not self._use_tune:
        # configure logging module if tuning is not active, else do it within the train method
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

        self._do_some_sanity_checks()


    @classmethod
    def from_args(cls, args, mode=None, options=''):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args, unknown = args.parse_known_args()

        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        if args.resume is not None:
            resume = Path(args.resume)
            if args.tune:
                cfg_path = resume.parent.parent / 'config.json'
            else:
                cfg_path = resume.parent / 'config.json'
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_path = Path(os.path.join(get_project_root(), args.config))


        config = read_json(cfg_path)
        if args.config and resume:
            # update new config for fine-tuning
            config.update(read_json(args.config))

        if args.config and args.seed:
            # Append the manual set seed to the config
            config['SEED'] = args.seed

        # parse custom cli options into dictionary
        modification = {opt.target: getattr(args, _get_opt_name(opt.flags)) for opt in options}
        return cls(config, resume, modification, mode, args.tune)

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self[name]['type']  # e.g., MnistDataLoader
        module_args = dict(self[name]['args'])  # e.g., {'data_dir': 'data/', 'batch_size': 4, ..., 'seq_len': 72000}
        # kwargs, i.e., single_batch for the data_loader
        # assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        # Gets a named attribute (here named "module_name") from an object (here from the given data-loader module)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity,
                                                                                       self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def resume(self):
        return self._resume

    @resume.setter
    def resume(self, value):
        self._resume = value

    @property
    def save_dir(self):
        return self._save_dir

    @save_dir.setter
    def save_dir(self, value):
        self._save_dir = value

    @property
    def log_dir(self):
        return self._log_dir

    @log_dir.setter
    def log_dir(self, value):
        self._log_dir = value

    @property
    def test_output_dir(self):
        return self._test_output_dir

    @test_output_dir.setter
    def test_output_dir(self, value):
        self._test_output_dir = value

    @property
    def use_tune(self):
        return self._use_tune

    def _do_some_sanity_checks(self):
        if self.config["loss"]["type"] == "BCE_with_logits" or self.config["loss"]["type"] == "balanced_BCE_with_logits":
            assert self.config["arch"]["args"]["multi_label_training"] \
                   and not self.config["arch"]["args"]["apply_final_activation"] \
                   and not self.config["metrics"]["additional_metrics_args"]["sigmoid_probs"] \
                   and not self.config["metrics"]["additional_metrics_args"]["log_probs"] \
                   and self.config["metrics"]["additional_metrics_args"]["logits"], "The used loss does not " \
                                                                                    "fit to the rest of the " \
                                                                                    "configuration"
        elif self.config["loss"]["type"] == "BCE":
            assert self.config["arch"]["args"]["multi_label_training"] \
                   and self.config["arch"]["args"]["apply_final_activation"] \
                   and self.config["metrics"]["additional_metrics_args"]["sigmoid_probs"] \
                   and not self.config["metrics"]["additional_metrics_args"]["log_probs"] \
                   and not self.config["metrics"]["additional_metrics_args"]["logits"], "The used loss does not " \
                                                                                        "fit to the rest of the " \
                                                                                        "configuration "
        elif self.config["loss"]["type"] == "nll_loss":
            assert not self.config["arch"]["args"]["multi_label_training"] \
                   and self.config["arch"]["args"]["apply_final_activation"] \
                   and not self.config["metrics"]["additional_metrics_args"]["sigmoid_probs"] \
                   and self.config["metrics"]["additional_metrics_args"]["log_probs"] \
                   and not self.config["metrics"]["additional_metrics_args"]["logits"], "The used loss does not " \
                                                                                        "fit to the rest of the " \
                                                                                        "configuration "
        elif self.config["loss"]["type"] == "cross_entropy_loss" or self.config["loss"]["type"] == "balanced_cross_entropy":
            assert not self.config["arch"]["args"]["multi_label_training"] \
                   and not self.config["arch"]["args"]["apply_final_activation"] \
                   and not self.config["metrics"]["additional_metrics_args"]["sigmoid_probs"] \
                   and not self.config["metrics"]["additional_metrics_args"]["log_probs"] \
                   and self.config["metrics"]["additional_metrics_args"]["logits"], "The used loss does not " \
                                                                                    "fit to the rest of the " \
                                                                                    "configuration "

        assert (self.config["metrics"]["additional_metrics_args"]["sigmoid_probs"] ^
                self.config["metrics"]["additional_metrics_args"]["log_probs"]
                ^ self.config["metrics"]["additional_metrics_args"]["logits"]) and not \
                   (self.config["metrics"]["additional_metrics_args"]["sigmoid_probs"]
                    and self.config["metrics"]["additional_metrics_args"]["log_probs"]
                    and self.config["metrics"]["additional_metrics_args"]["logits"]), \
            "Exactly one of the three must be true"


# helper functions to update config dict with custom cli options
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            if isinstance(v, Path):
                v = v.__str__()
            _set_by_path(config, k, v)
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
