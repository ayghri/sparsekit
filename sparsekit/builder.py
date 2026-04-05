# Copyright (c) 2026 - Ayoub Ghriss & Contributors
# Licensed under CC BY-NC 4.0
# (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
# Non-commercial use only; contact us for commercial licensing.
"""Fluent API for building block/scope sparsity hierarchies.

Example::

    builder = (
        SparsityBuilder()
        .add_block(param_u, (2, 2, 2, 2), name="U")
        .add_block(param_v, (2, 2, 2, 2), name="V")
        .add_scope("U", scope_shape=(1, 1), name="gU")
        .add_scope("V", scope_shape=(1, 4), name="gV")
        .couple_scopes(
            ["gU", "gV"],
            orders=[(0,1,2,3), (1,0,2,3)],
            name="UV",
        )
    )
    coupling = builder.get_scope("UV")
"""

from typing import Dict, List, Tuple, Iterable
from torch.nn import Parameter

from .view import View
from .block import BlockSpec
from .block import BlockCoupling
from .scope import ScopeSpec
from .scope import ScopeCoupling


class SparsityBuilder:
    """Fluent builder for constructing BlockSpec/ScopeSpec hierarchies.

    All mutating methods return ``self`` for method chaining.
    """

    def __init__(self):
        self._blocks: Dict[str, BlockSpec] = {}
        self._g_couplings: Dict[str, BlockCoupling] = {}
        self._scopes: Dict[str, ScopeSpec] = {}
        self._s_couplings: Dict[str, ScopeCoupling] = {}

    def add_block(
        self, param: Parameter, block_shape: Tuple[int, ...], name: str
    ):
        """Register a single parameter with its block decomposition."""
        assert name not in self._blocks
        view = View.from_existing(param)
        # pylint: disable=abstract-class-instantiated
        self._blocks[name] = BlockSpec(view, block_shape, name=name)
        return self

    def couple_blocks(
        self, block_names: List[str], orders: List[Tuple[int, ...]], name: str
    ):
        """Create a BlockCoupling from previously added blocks."""
        self._g_couplings[name] = BlockCoupling(
            [self._blocks[n] for n in block_names], orders, name=name
        )
        return self

    def get_block(self, name: str) -> BlockCoupling | BlockSpec:
        """Retrieve a BlockSpec or BlockCoupling by name."""
        if name in self._blocks:
            return self._blocks[name]
        return self._g_couplings[name]

    def add_scope(
        self, block_name: str, scope_shape: Tuple[int, ...], name: str
    ):
        """Add a ScopeSpec over an existing block or coupling."""
        assert name not in self._scopes
        self._scopes[name] = ScopeSpec(
            self.get_block(block_name),
            shape=scope_shape,
            name=name,
        )
        return self

    def couple_scopes(
        self,
        scope_names: List[str],
        orders: List[Tuple[int, ...]],
        name: str,
    ):
        """Create a ScopeCoupling from previously added scopes.

        Scopes are consumed (popped) from the builder when coupled.
        """
        assert name not in self._s_couplings
        self._s_couplings[name] = ScopeCoupling(
            [self._scopes.pop(n) for n in scope_names],
            orders,
            name=name,
        )
        return self

    def get_scope(self, name: str) -> ScopeSpec | ScopeCoupling:
        """Retrieve a ScopeSpec or ScopeCoupling by name."""
        if name in self._scopes:
            return self._scopes[name]
        return self._s_couplings[name]

    def get_all_scopes(self) -> Iterable[ScopeSpec | ScopeCoupling]:
        """Return all uncoupled scopes and scope couplings."""
        return list(self._scopes.values()) + list(
            self._s_couplings.values()
        )
