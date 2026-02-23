from typing import Dict, List, Tuple, Iterable
from torch.nn import Parameter

from .blocks import BlockSpec
from .blocks import BlockCoupling
from .groups import GroupSpec
from .groups import GroupCoupling

class SparsityBuilder:
    def __init__(self):
        self._blocks: Dict[str, BlockSpec] = {}
        self._b_couplings: Dict[str, BlockCoupling] = {}
        self._groups: Dict[str, GroupSpec] = {}
        self._g_couplings: Dict[str, GroupCoupling] = {}

    def add_block(
        self, param: Parameter, block_shape: Tuple[int, ...], name: str
    ):
        assert name not in self._blocks
        self._blocks[name] = BlockSpec(param, block_shape, name=name)
        return self

    def couple_blocks(
        self, block_names: List[str], orders: List[Tuple[int, ...]], name: str
    ):
        self._b_couplings[name] = BlockCoupling(
            [self._blocks[n] for n in block_names], orders, name=name
        )
        return self

    def get_block(self, name) -> BlockCoupling | BlockSpec:
        if name in self._blocks:
            return self._blocks[name]
        return self._b_couplings[name]

    def add_group(self, block_name: str, group_shape, name: str):
        assert name not in self._groups
        self._groups[name] = GroupSpec(
            self.get_block(block_name),
            group_shape=group_shape,
            name=name,
        )
        return self

    def couple_groups(
        self, group_names: List[str], orders: List[Tuple[int, ...]], name: str
    ):
        assert name not in self._g_couplings
        self._g_couplings[name] = GroupCoupling(
            [self._groups.pop(n) for n in group_names], orders, name=name
        )
        return self

    def get_group(self, name) -> GroupSpec | GroupCoupling:
        if name in self._groups:
            return self._groups[name]
        return self._g_couplings[name]

    def get_all_groups(self) -> Iterable[GroupSpec | GroupCoupling]:
        return list(self._groups.values()) + list(self._g_couplings.values())
