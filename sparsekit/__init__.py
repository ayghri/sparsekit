"""
Copyright (c) 2025 Ayoub Ghriss and contributors
Licensed under CC BY-NC 4.0 (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
Non-commercial use only; contact us for commercial licensing.

SparseKit: Block-structured sparse tensor operations.

"""

from . import blocks
from . import groups
from . import linalg
from . import builder
from .blocks import BlockSpec, BlockCoupling
from .groups import GroupSpec, GroupCoupling
from . import utils

__all__ = [
    "blocks",
    "groups",
    "linalg",
    "builder",
    "BlockSpec",
    "BlockCoupling",
    "GroupSpec",
    "GroupCoupling",
    "utils",
]
