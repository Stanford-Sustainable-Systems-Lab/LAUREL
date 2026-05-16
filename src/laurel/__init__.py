"""EPA Phase III"""

__version__ = "0.1"

import warnings

warnings.filterwarnings(
    "ignore",
    message="FNV hashing is not implemented in Numba",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
)
