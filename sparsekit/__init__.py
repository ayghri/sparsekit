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
from .views import BlockView
from . import utils
from .pruners import obs, quant, nvquant
from .pruners.obs import StructuredOBS
from .pruners.quant import quantize_obs, mxfp4_quantize
from .pruners.nvquant import nvfp4_quantize, quantize_nvfp4_obs

__all__ = [
    "blocks",
    "groups",
    "linalg",
    "builder",
    "obs",
    "BlockSpec",
    "BlockCoupling",
    "GroupSpec",
    "GroupCoupling",
    "BlockView",
    "StructuredOBS",
    "quantize_obs",
    "mxfp4_quantize",
    "quant",
    "nvquant",
    "nvfp4_quantize",
    "quantize_nvfp4_obs",
    "utils",
]
