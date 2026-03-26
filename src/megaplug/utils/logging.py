"""Logging utilities for suppressing verbose third-party output during pipeline runs.

Some libraries (e.g. ``osmium``, ``dask``) emit large volumes of INFO or
WARNING messages that obscure meaningful pipeline output.  This module provides
a context manager to temporarily silence the root logger at a configurable
level.
"""

import logging


class SuppressLogs:
    """Context manager that globally disables logging up to a given severity level.

    Saves and restores the root logger's disabled level on entry and exit,
    so temporary suppression does not leak past the ``with`` block.

    Args:
        level: Logging level at and below which messages are suppressed.
            Defaults to ``logging.CRITICAL`` (suppress everything).

    Example::

        with SuppressLogs(logging.WARNING):
            noisy_library_call()  # INFO and WARNING messages suppressed
    """

    def __init__(self, level=logging.CRITICAL):
        self.level = level

    def __enter__(self):
        self.original_level = logging.root.manager.disable
        logging.disable(self.level)

    def __exit__(self, exc_type, exc_value, traceback):
        logging.disable(self.original_level)
