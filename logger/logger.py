import logging
import logging.config
import os
from pathlib import Path

from utils import get_project_root, read_json

'''
INFORMATION about logging (see https://docs.python.org/3/howto/logging.html#logging-basic-tutorial)
-   Log event information is passed between loggers, handlers, filters and formatters in a LogRecord instance.
-   Logging is performed by calling methods on instances of the Logger class (hereafter called loggers). 
    Each instance has a name, and they are conceptually arranged in a namespace hierarchy using dots as separators. 
    A good convention to use when naming loggers is to use a module-level logger, in each module which uses logging, 
    named as follows: logger = logging.getLogger(__name__)
-   It is, of course, possible to log messages to different destinations. 
    Destinations are served by handler classes.
    

Loggers:
- Configuration:    Logger.setLevel() , Logger.addHandler(), Logger.removeHandler(), 
                    Logger.addFilter(), Logger.removeFilter()
- Message sending:  Logger.debug(), Logger.info(), Logger.warning(), Logger.error(), Logger.critical(), 
                    Logger.exception(), Logger.log() 
- getLogger() returns a reference to a logger instance with the specified name if it is provided, or root if not.

Handlers:
- Responsible for dispatching the appropriate log messages (based on severity) to the handler’s specified destination. 
- Logger objects can add zero or more handler objects to themselves with an addHandler() method. 
- Important methods: setLevel(), setFormatter(), addFilter(), removeFilter() 

Formatters:
- Configure the final order, structure, and contents of the log message
- The constructor takes three optional arguments – a message format string, a date format string and a style indicator.

Configuration possibilities for logging:
1) Creating loggers, handlers, and formatters explicitly using Python code 
2) Creating a logging config file and reading it using the fileConfig() function.
3) Creating a dictionary of configuration information and passing it to the dictConfig() function.
'''


def setup_logging(save_dir, log_config_path='logger/logger_config.json', default_level=logging.INFO):
    """
    Setup logging configuration
    """
    log_config_path = Path(os.path.join(get_project_root(), log_config_path))
    if log_config_path.is_file():
        # Important: The config file has to follow the following schema:
        # https://docs.python.org/3/library/logging.config.html#logging-config-dictschema
        config = read_json(log_config_path)

        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            # Handlers send the log records (created by loggers) to the appropriate destination.
            if 'filename' in handler and save_dir is not None:
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config_path))
        logging.basicConfig(level=default_level)


def update_logging_setup_for_tune(new_save_path):
    setup_logging(new_save_path)
