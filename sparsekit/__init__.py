# Copyright (c) 2026 - Ayoub Ghriss & Contributors
# Licensed under CC BY-NC 4.0
# (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
# Non-commercial use only; contact us for commercial licensing.
"""SparseKit: Group-structured sparse tensor operations."""

from . import block
from . import scope
from . import builder
from . import tensor_ops
from . import linalg
from .block import BlockSpec, BlockCoupling
from .scope import ScopeSpec, ScopeCoupling
from .view import View
from .viz import draw_layout
from .pruners import obs, obd
from .pruners.obs import StructuredOBS

__all__ = [
    "block",
    "scope",
    "builder",
    "tensor_ops",
    "linalg",
    "obs",
    "obd",
    "BlockSpec",
    "BlockCoupling",
    "ScopeSpec",
    "ScopeCoupling",
    "View",
    "StructuredOBS",
    "draw_layout",
]
