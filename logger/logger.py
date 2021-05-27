import logging
import logging.config
from pathlib import Path
from utils import get_project_root, read_json
import os


def setup_logging(save_dir, log_config_path='logger/logger_config.json', default_level=logging.INFO):
    """
    Setup logging configuration
    """
    log_config_path = Path(os.path.join(get_project_root(), log_config_path))
    if log_config_path.is_file():
        config = read_json(log_config_path)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config_path))
        logging.basicConfig(level=default_level)
