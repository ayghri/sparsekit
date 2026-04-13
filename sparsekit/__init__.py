# Copyright (c) 2026 - Ayoub Ghriss & Contributors
# Licensed under CC BY-NC 4.0
# (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
# Non-commercial use only; contact us for commercial licensing.
"""SparseKit: Structured sparsity specification and pruning."""

from . import block
from . import scope
from . import builder
from . import tensor_ops
from . import linalg
from .block import BlockSpec, BlockCoupling
from .scope import ScopeSpec, ScopeCoupling
from .view import View
from .pruners import obs, obd, sparsegpt
from .pruners.obs import StructuredOBS
from .pruners.obd import magnitude, obd as obd_prune
from .pruners.sparsegpt import (
    sparsegpt as sparsegpt_prune,
    sparsegpt_coupled_24,
    sparsegpt_block16,
)
from .pruners import compute_hessian, output_error

__all__ = [
    "block",
    "scope",
    "builder",
    "tensor_ops",
    "linalg",
    "obs",
    "obd",
    "sparsegpt",
    "BlockSpec",
    "BlockCoupling",
    "ScopeSpec",
    "ScopeCoupling",
    "View",
    "StructuredOBS",
    "magnitude",
    "obd_prune",
    "sparsegpt_prune",
    "sparsegpt_coupled_24",
    "sparsegpt_block16",
    "compute_hessian",
    "output_error",
]
