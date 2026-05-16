import logging


class SuppressDistributedFilter(logging.Filter):
    """Rejects log records whose logger name starts with 'distributed'.

    Attached to Kedro's Rich handler in conf/logging.yml so that Dask worker
    startup/shutdown banners are suppressed regardless of how the records arrive
    in the main process (direct logging calls or IPC forwarding from subprocesses).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name.startswith("distributed"):
            return record.levelno >= logging.WARNING
        return True
