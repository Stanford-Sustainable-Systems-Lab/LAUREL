import logging


class SuppressLogs:
    def __init__(self, level=logging.CRITICAL):
        self.level = level

    def __enter__(self):
        self.original_level = logging.root.manager.disable

        logging.disable(self.level)

    def __exit__(self, exc_type, exc_value, traceback):
        logging.disable(self.original_level)
