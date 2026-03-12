"""Fluent API for building block/group sparsity hierarchies.

Example::

    builder = (
        SparsityBuilder()
        .add_block(param_u, (2, 2, 2, 2), name="U")
        .add_block(param_v, (2, 2, 2, 2), name="V")
        .add_group("U", group_shape=(1, 1), name="gU")
        .add_group("V", group_shape=(1, 4), name="gV")
        .couple_groups(["gU", "gV"], orders=[(0,1,2,3), (1,0,2,3)], name="UV")
    )
    coupling = builder.get_group("UV")
"""

from typing import Dict, List, Tuple, Iterable
from torch.nn import Parameter

from .block import BlockSpec
from .block import BlockCoupling
from .group import GroupSpec
from .group import GroupCoupling


class SparsityBuilder:
    """Fluent builder for constructing BlockSpec/GroupSpec hierarchies.

    All mutating methods return ``self`` for method chaining.
    """

    def __init__(self):
        self._blocks: Dict[str, BlockSpec] = {}
        self._b_couplings: Dict[str, BlockCoupling] = {}
        self._groups: Dict[str, GroupSpec] = {}
        self._g_couplings: Dict[str, GroupCoupling] = {}

    def add_block(
        self, param: Parameter, block_shape: Tuple[int, ...], name: str
    ):
        """Register a single parameter with its block decomposition."""
        assert name not in self._blocks
        self._blocks[name] = BlockSpec(param, block_shape, name=name)
        return self

    def couple_blocks(
        self, block_names: List[str], orders: List[Tuple[int, ...]], name: str
    ):
        """Create a BlockCoupling from previously added blocks."""
        self._b_couplings[name] = BlockCoupling(
            [self._blocks[n] for n in block_names], orders, name=name
        )
        return self

    def get_block(self, name: str) -> BlockCoupling | BlockSpec:
        """Retrieve a BlockSpec or BlockCoupling by name."""
        if name in self._blocks:
            return self._blocks[name]
        return self._b_couplings[name]

    def add_group(self, block_name: str, group_shape: Tuple[int, ...], name: str):
        """Add a GroupSpec over an existing block or coupling."""
        assert name not in self._groups
        self._groups[name] = GroupSpec(
            self.get_block(block_name),
            shape=group_shape,
            name=name,
        )
        return self

    def couple_groups(
        self, group_names: List[str], orders: List[Tuple[int, ...]], name: str
    ):
        """Create a GroupCoupling from previously added groups.

        Groups are consumed (popped) from the builder when coupled.
        """
        assert name not in self._g_couplings
        self._g_couplings[name] = GroupCoupling(
            [self._groups.pop(n) for n in group_names], orders, name=name
        )
        return self

    def get_group(self, name: str) -> GroupSpec | GroupCoupling:
        """Retrieve a GroupSpec or GroupCoupling by name."""
        if name in self._groups:
            return self._groups[name]
        return self._g_couplings[name]

    def get_all_groups(self) -> Iterable[GroupSpec | GroupCoupling]:
        """Return all uncoupled groups and group couplings."""
        return list(self._groups.values()) + list(self._g_couplings.values())
