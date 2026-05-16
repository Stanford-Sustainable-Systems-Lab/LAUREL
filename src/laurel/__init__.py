"""EPA Phase III"""

__version__ = "0.1"

import warnings

warnings.filterwarnings(
    "ignore",
    message="FNV hashing is not implemented in Numba",
    category=UserWarning,
)
