import pytest
import torch
from torch.nn import Parameter

from sparsekit.blocks import BlockSpec
from sparsekit.utils import  CouplingError
from sparsekit.groups import GroupSpec, GroupCoupling


@pytest.fixture
def simple_block_spec():
    # 4x4 tensor, 2x2 blocks -> block_grid_shape (2,2)
    W = torch.arange(16.0).view(4, 4)
    p = Parameter(W.clone())
    return BlockSpec(p, block_shape=(2, 2))


class TestGroupSpecInit:
    def test_init_default_group_shape(self, simple_block_spec):
        g = GroupSpec(simple_block_spec, group_shape=())
        assert g.group_shape == simple_block_spec.grid_shape
        assert g.grid_shape == (1, 1)
        assert g.num_groups == 1
        assert g.group_numel == g.group_shape[0] * g.group_shape[1]

    def test_init_explicit_group_shape(self, simple_block_spec):
        g = GroupSpec(simple_block_spec, group_shape=(1, 2))
        assert g.group_shape == (1, 2)
        assert g.grid_shape == (2, 1)
        assert g.num_groups == 2 * 1
        assert g.group_numel == 1 * 2

    def test_init_group_shape_mismatch_raises(self, simple_block_spec):
        with pytest.raises(ValueError):
            GroupSpec(simple_block_spec, group_shape=(2, 2, 2))

    def test_init_group_shape_not_divisible_raises(self, simple_block_spec):
        with pytest.raises(ValueError):
            GroupSpec(simple_block_spec, group_shape=(3, 1))


class TestGroupSpecViews:
    def test_block_to_group_and_back_identity(self, simple_block_spec):
        g = GroupSpec(simple_block_spec, group_shape=())
        # use block norms as representative per-block values
        block_vals = simple_block_spec.block_norms(None)
        grouped = g.block_to_group(block_vals, reorder=False)
        # grouped shape should be (1,2,1,2)
        assert grouped.view(-1).numel() == g.num_groups * g.group_numel

        # Round-trip via group_to_block
        group_vals = torch.ones(g.grid_shape)
        blocks_back = g.group_to_block(group_vals)
        assert tuple(blocks_back.shape) == tuple(simple_block_spec.grid_shape)


class TestGroupSpecHardThreshold:
    def test_hard_threshold_with_explicit_group_thresholds(
        self, simple_block_spec
    ):
        g = GroupSpec(simple_block_spec, group_shape=())
        # two 2x2 blocks, make first big, second small
        simple_block_spec.set_data(
            torch.tensor(
                [
                    [10.0, 10.0, 1.0, 1.0],
                    [10.0, 10.0, 1.0, 1.0],
                    [10.0, 10.0, 1.0, 1.0],
                    [10.0, 10.0, 1.0, 1.0],
                ]
            )
        )
        # group over both blocks together (only one group)
        thresholds = torch.zeros(g.grid_shape)
        before = simple_block_spec.data.clone()
        g.hard_threshold(thresholds=thresholds)
        # threshold=0 leaves everything (since norms >= 0)
        assert torch.allclose(simple_block_spec.data, before)

    def test_hard_threshold_with_sparsity(self, simple_block_spec):
        # group_shape=(1,1) -> each block its own group
        g = GroupSpec(simple_block_spec, group_shape=(-1, 1))
        # Make one block large, one block very small
        data = torch.zeros(4, 4)
        data[0:2, 0:2] = 10.0
        data[2:4, 2:4] = 0.1
        simple_block_spec.set_data(data)
        # sparsity=0.5 -> keep half the groups => one group; the small one should go
        g.hard_threshold(sparsity=0.5)

        # One block should remain non-zero, one should be all zeros
        block_norms = simple_block_spec.block_norms(None)
        assert (block_norms == 0).sum() == 2
        assert (block_norms > 0).sum() == 2


class TestGroupSpecSoftThreshold:
    def test_scales_like_block_soft_threshold(self, simple_block_spec):
        # Make blocks of ones so norm and scaling are easy to reason about
        simple_block_spec.set_data(torch.ones_like(simple_block_spec.data))
        # One group over all blocks
        g = GroupSpec(simple_block_spec, group_shape=())
        lambdas = torch.ones(g.grid_shape)
        conditioners = torch.ones_like(simple_block_spec.data)
        before = simple_block_spec.data.clone()
        g.soft_threshold(
            lambdas, conditioners={simple_block_spec: conditioners}, eps=1e-12
        )
        after = simple_block_spec.data
        # Should shrink but not zero everything
        assert not torch.allclose(after, before)
        assert torch.any(after != 0.0)

    def test_scale_flag(self, simple_block_spec):
        simple_block_spec.set_data(torch.ones_like(simple_block_spec.data))
        g = GroupSpec(simple_block_spec, group_shape=())
        lambdas = torch.ones(g.grid_shape)
        conditioners = torch.ones_like(simple_block_spec.data)

        # With scale=False vs True we should get different results
        g.soft_threshold(
            lambdas,
            conditioners={simple_block_spec: conditioners},
            eps=1e-12,
            scale=False,
        )
        no_scale = simple_block_spec.data.clone()

        simple_block_spec.set_data(torch.ones_like(simple_block_spec.data))
        g.soft_threshold(
            lambdas,
            conditioners={simple_block_spec: conditioners},
            eps=1e-12,
            scale=True,
        )
        with_scale = simple_block_spec.data

        assert not torch.allclose(no_scale, with_scale)

    def test_adam(self):
        v = Parameter(torch.ones(4, 2, 2))
        v.data[-1].mul_(0.9)

        h = torch.stack(
            [
                torch.ones(2, 2),
                torch.tensor([[0.25, 0.5], [1.0, 2.0]]),
                torch.tensor(
                    [[0.49671415, 0.1382643], [0.64768854, 1.52302986]]
                ),
                torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
            ]
        )

        spec = BlockSpec(v, block_shape=(1, 2, 2))
        g = GroupSpec(spec, group_shape=(2,))
        thresholds = torch.tensor([0.5, 1.0]).reshape(g.grid_shape)
        expected = torch.stack(
            [
                torch.full((2, 2), 0.75),
                v.data[1] * h[1] / (h[1] + 0.3830713728),
                v.data[2] * h[2] / (h[2] + 1.6383774184),
                torch.zeros((2, 2)),
            ]
        )
        g.soft_threshold(thresholds, conditioners={spec: h}, max_iter=50)

        assert torch.allclose(spec.data, expected)


class TestGroupCoupling:
    @pytest.fixture
    def params_uv(self):
        torch.manual_seed(0)
        U = Parameter(torch.randn(4, 8, 2, 2))
        V = Parameter(torch.randn(8, 16, 2, 2))
        return U, V

    @pytest.fixture
    def groups_uv(self, params_uv):
        U, V = params_uv
        # Match the __main__ example in groups.py
        block_u = BlockSpec(U, block_shape=(2, 2, 2, 2), name="U")
        group_u = GroupSpec(block_u, group_shape=(1, 1))

        block_v = BlockSpec(V, block_shape=(2, 2, 2, 2), name="V")
        group_v = GroupSpec(block_v, group_shape=(1, 4))

        return group_u, group_v

    @pytest.fixture
    def coupling(self, groups_uv):
        group_u, group_v = groups_uv
        # Full-rank orders over 4D grid_shape: swap first two dims, keep singletons
        return GroupCoupling([group_u, group_v], orders=[(0, 1, 2, 3), (1, 0, 2, 3)])

    def test_init_valid(self, groups_uv):
        group_u, group_v = groups_uv
        coupling = GroupCoupling(
            [group_u, group_v],
            orders=[(0, 1, 2, 3), (1, 0, 2, 3)],
        )
        assert len(coupling.groups) == 2
        assert len(coupling.orders) == 2
        # Group grids must match after permutation
        ref_order = coupling.orders[0]
        ref_shape = tuple(group_u.grid_shape[i] for i in ref_order)
        other_order = coupling.orders[1]
        other_shape = tuple(group_v.grid_shape[i] for i in other_order)
        assert ref_shape == other_shape

    def test_init_invalid_order_raises(self, groups_uv):
        group_u, group_v = groups_uv
        # Use incompatible orders that should fail the shape check in __post_init__
        with pytest.raises(CouplingError):
            GroupCoupling(
                [group_u, group_v],
                orders=[(0, 1, 2, 3), (0, 1, 2, 3)],
            )

    def test_grouped_block_norms_shape(self, coupling):
        # When values=None, GroupSpec.grouped_block_norms uses live data
        norms = coupling.grouped_block_norms(values=None)
        # Last dim is concatenated over groups; others should match grid_shape
        assert norms.shape[:-1] == coupling.grid_shape
        # There should be as many channels in the last dimension as total groups
        total_groups = sum(
            g.grouped_block_norms(None).shape[-1] for g in coupling.groups
        )
        assert norms.shape[-1] == total_groups

    def test_kth_largest_shape(self, coupling):
        # Use internal grouped scores with live data (values=None)
        grouped_scores = coupling.grouped_block_norms(values=None)
        k = 1
        from sparsekit.linalg import kth_largest

        thresholds = kth_largest(grouped_scores, k=k, dim=-1)
        # kth_largest over last dim should return a tensor with leading dims == grid_shape
        assert tuple(thresholds.shape) == coupling.grid_shape

    def test_hard_threshold_reduces_some_norms(self, coupling):
        before = [g.block.block_norms(None).clone() for g in coupling.groups]
        coupling.hard_threshold(num_nz=1)
        after = [g.block.block_norms(None).clone() for g in coupling.groups]

        # At least one block norm across all groups should have decreased or become zero
        assert any(torch.any(a <= b - 1e-6) for b, a in zip(before, after))

    def test_soft_threshold_reduces_param_norm(self, coupling):
        # Use simple conditioners (all ones) and a modest threshold
        group_thresholds = torch.full(coupling.grid_shape, 0.1)
        conditioners = {
            g.block: torch.ones_like(g.block.data) for g in coupling.groups
        }

        before = [
            torch.linalg.vector_norm(g.block.data) for g in coupling.groups
        ]
        coupling.soft_threshold(group_thresholds, conditioners=conditioners)
        after = [
            torch.linalg.vector_norm(g.block.data) for g in coupling.groups
        ]

        # Each parameter norm should not increase
        for b, a in zip(before, after):
            assert a <= b + 1e-6

    def test_soft_threshold_does_not_increase_block_norms(self, coupling):
        group_thresholds = torch.full(coupling.grid_shape, 0.1)
        conditioners = {
            g.block: torch.ones_like(g.block.data) for g in coupling.groups
        }

        before = [g.block.block_norms(None).clone() for g in coupling.groups]
        coupling.soft_threshold(group_thresholds, conditioners=conditioners)
        after = [g.block.block_norms(None).clone() for g in coupling.groups]

        for b, a in zip(before, after):
            assert torch.all(a <= b + 1e-6)
