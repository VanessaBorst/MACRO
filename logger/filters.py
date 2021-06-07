import logging.config


class FilterConsoleInfo(logging.Filter):
    def __init__(self, max_line_length=2):
        self._max_line_len = max_line_length

    def filter(self, record):
        # Filter out all logging records containing long strings,
        # i.e., strings whose line length exceeds the maximum defined in the config (e.g. the confusion matrices)
        return len(record.msg.splitlines()) <= self._max_line_len if isinstance(record.msg, str) else True