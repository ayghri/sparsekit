import pytest
import torch
from torch.nn import Parameter

from sparsekit.block import BlockSpec
from sparsekit.block import CouplingError
from sparsekit.scope import ScopeSpec, ScopeCoupling


@pytest.fixture
def simple_block_spec():
    # 4x4 tensor, 2x2 groups -> block_grid_shape (2,2)
    W = torch.arange(16.0).view(4, 4)
    p = Parameter(W.clone())
    return BlockSpec(p, shape=(2, 2))


class TestScopeSpecInit:
    def test_init_default_block_shape(self, simple_block_spec):
        g = ScopeSpec(simple_block_spec, shape=())
        assert g.shape == simple_block_spec.grid_shape
        assert g.grid_shape == (1, 1)
        assert g.numscp() == 1
        assert g.numblk_per_scope() == g.shape[0] * g.shape[1]

    def test_init_explicit_block_shape(self, simple_block_spec):
        g = ScopeSpec(simple_block_spec, shape=(1, 2))
        assert g.shape == (1, 2)
        assert g.grid_shape == (2, 1)
        assert g.numscp() == 2 * 1
        assert g.numblk_per_scope() == 1 * 2

    def test_init_block_shape_mismatch_raises(self, simple_block_spec):
        with pytest.raises(ValueError):
            ScopeSpec(simple_block_spec, shape=(2, 2, 2))

    def test_init_block_shape_not_divisible_raises(self, simple_block_spec):
        with pytest.raises(ValueError):
            ScopeSpec(simple_block_spec, shape=(3, 1))


class TestScopeSpecViews:
    def test_block_to_scope_and_back_identity(self, simple_block_spec):
        g = ScopeSpec(simple_block_spec, shape=())
        # use group norms as representative per-group values (non-uniform)
        block_vals = simple_block_spec.norms(None)
        grouped = g.block_to_scope(block_vals, reorder=False)
        assert grouped.view(-1).numel() == g.numscp() * g.numblk_per_scope()
        # Values are only rearranged, not changed
        assert torch.allclose(
            grouped.view(-1).sort().values, block_vals.view(-1).sort().values
        )

        # scope_to_block broadcasts per-scope scalars back to block grid
        group_vals = torch.ones(g.grid_shape)
        blocks_back = g.scope_to_block(group_vals)
        assert tuple(blocks_back.shape) == tuple(simple_block_spec.grid_shape)
        assert torch.allclose(blocks_back, torch.ones(simple_block_spec.grid_shape))


class TestScopeSpecHardThreshold:
    def test_hard_threshold_with_explicit_block_thresholds(self, simple_block_spec):
        g = ScopeSpec(simple_block_spec, shape=())
        # two 2x2 groups, make first big, second small
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
        # block over both groups together (only one block)
        thresholds = torch.zeros(g.grid_shape)
        before = simple_block_spec.data.clone()
        g.hard_threshold(thresholds=thresholds)
        # threshold=0 leaves everything (since norms >= 0)
        assert torch.allclose(simple_block_spec.data, before)

    def test_hard_threshold_with_sparsity(self, simple_block_spec):
        # ScopeSpec with shape=(-1,1): entire column of blocks forms one scope
        g = ScopeSpec(simple_block_spec, shape=(-1, 1))
        # Make one group large, one group very small
        data = torch.zeros(4, 4)
        data[0:2, 0:2] = 10.0
        data[2:4, 2:4] = 0.1
        simple_block_spec.set_data(data)
        # sparsity=0.5 -> keep half the blocks => one block; the small one should go

        nnz = g.sparsity_to_nnz(0.5)
        g.hard_threshold(nnz=nnz)

        # One group should remain non-zero, one should be all zeros
        block_norms = simple_block_spec.norms(None)
        assert (block_norms == 0).sum() == 2
        assert (block_norms > 0).sum() == 2

        # Verify WHICH blocks survived: large-norm blocks win their scopes
        # Scope (col 0): blocks (0,0)=norm 20 vs (1,0)=norm 0 → (0,0) survives
        assert block_norms[0, 0] > 0, "Large block [0:2,0:2] (norm=20) should survive"
        assert block_norms[1, 0] == 0, "Zero block [2:4,0:2] (norm=0) should be pruned"
        # Scope (col 1): blocks (0,1)=norm 0 vs (1,1)=norm 0.2 → (1,1) survives
        assert (
            block_norms[1, 1] > 0
        ), "Small block [2:4,2:4] (norm=0.2) should survive (largest in scope)"
        assert block_norms[0, 1] == 0, "Zero block [0:2,2:4] (norm=0) should be pruned"


class TestScopeSpecSoftThreshold:
    def test_scales_like_block_soft_threshold(self, simple_block_spec):
        # Make groups of ones so norm and scaling are easy to reason about
        simple_block_spec.set_data(torch.ones_like(simple_block_spec.data))
        # One block over all groups
        g = ScopeSpec(simple_block_spec, shape=())
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
        # With H=ones, lambda=1, per-block norm=2:
        # Adam: mu/(1+mu)*2 = 1 → mu=1 → scale=0.5
        assert torch.allclose(after, torch.full_like(after, 0.5))

    def test_scale_flag(self, simple_block_spec):
        simple_block_spec.set_data(torch.ones_like(simple_block_spec.data))
        g = ScopeSpec(simple_block_spec, shape=())
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
        # scale=False: per-block threshold=1, per-block norm=2, H=1
        # Adam: mu/(1+mu)*2=1 → mu=1 → scale=0.5
        assert torch.allclose(no_scale, torch.full_like(no_scale, 0.5))
        # scale=True: threshold *= sqrt(block_numel)=sqrt(4)=2, norm=2 → threshold>=norm → zeroed
        assert torch.allclose(with_scale, torch.zeros_like(with_scale))

    def test_adam(self):
        v = Parameter(torch.ones(4, 2, 2))
        v.data[-1].mul_(0.9)

        h = torch.stack(
            [
                torch.ones(2, 2),
                torch.tensor([[0.25, 0.5], [1.0, 2.0]]),
                torch.tensor([[0.49671415, 0.1382643], [0.64768854, 1.52302986]]),
                torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
            ]
        )

        spec = BlockSpec(v, shape=(1, 2, 2))
        g = ScopeSpec(spec, shape=(2,))
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


class TestScopeCoupling:
    @pytest.fixture
    def params_uv(self):
        torch.manual_seed(0)
        U = Parameter(torch.randn(4, 8, 2, 2))
        V = Parameter(torch.randn(8, 16, 2, 2))
        return U, V

    @pytest.fixture
    def scope_uv(self, params_uv):
        U, V = params_uv
        # Match the __main__ example in blocks.py
        block_u = BlockSpec(U, shape=(2, 2, 2, 2), name="U")
        scope_u = ScopeSpec(block_u, shape=(1, 1))

        block_v = BlockSpec(V, shape=(2, 2, 2, 2), name="V")
        scope_v = ScopeSpec(block_v, shape=(1, 4))

        return scope_u, scope_v

    @pytest.fixture
    def coupling(self, scope_uv):
        scope_u, scope_v = scope_uv
        # Full-rank orders over 4D grid_shape: swap first two dims, keep singletons
        return ScopeCoupling([scope_u, scope_v], orders=[(0, 1, 2, 3), (1, 0, 2, 3)])

    def test_init_valid(self, scope_uv):
        scope_u, scope_v = scope_uv
        coupling = ScopeCoupling(
            [scope_u, scope_v],
            orders=[(0, 1, 2, 3), (1, 0, 2, 3)],
        )
        assert len(coupling.scopes) == 2
        assert len(coupling.orders) == 2
        # Block grids must match after permutation
        ref_order = coupling.orders[0]
        ref_shape = tuple(scope_u.grid_shape[i] for i in ref_order)
        other_order = coupling.orders[1]
        other_shape = tuple(scope_v.grid_shape[i] for i in other_order)
        assert ref_shape == other_shape

    def test_init_invalid_order_raises(self, scope_uv):
        scope_u, scope_v = scope_uv
        # Use incompatible orders that should fail the shape check in __post_init__
        with pytest.raises(CouplingError):
            ScopeCoupling(
                [scope_u, scope_v],
                orders=[(0, 1, 2, 3), (0, 1, 2, 3)],
            )

    def test_block_norms_shape(self, coupling):
        # When values=None, ScopeSpec.block_norms uses live data
        norms = coupling.block_norms(values=None)
        # Last dim is concatenated over blocks; others should match grid_shape
        assert norms.shape[:-1] == coupling.grid_shape
        # There should be as many channels in the last dimension as total blocks
        total_blocks = sum(g.block_norms(None).shape[-1] for g in coupling.scopes)
        assert norms.shape[-1] == total_blocks

    def test_kth_largest_shape(self, coupling):
        # Use internal grouped scores with live data (values=None)
        grouped_scores = coupling.block_norms(values=None)
        k = 1
        from sparsekit.tensor_ops import kth_largest

        thresholds = kth_largest(grouped_scores, k=k, dim=-1)
        # kth_largest over last dim should return a tensor with leading dims == grid_shape
        assert tuple(thresholds.shape) == coupling.grid_shape

    def test_hard_threshold_reduces_some_norms(self, coupling):
        before = [g.block.norms(None).clone() for g in coupling.scopes]
        coupling.hard_threshold(nnz=1)
        after = [g.block.norms(None).clone() for g in coupling.scopes]

        # At least one group norm across all blocks should have decreased or become zero
        assert any(torch.any(a <= b - 1e-6) for b, a in zip(before, after))
        # Some blocks must be exactly zero (pruned)
        assert any(torch.any(a == 0) for a in after)

    def test_soft_threshold_reduces_param_norm(self, coupling):
        # Use simple conditioners (all ones) and a modest threshold
        block_thresholds = torch.full(coupling.grid_shape, 0.1)
        conditioners = {g.block: torch.ones_like(g.block.data) for g in coupling.scopes}

        before = [torch.linalg.vector_norm(g.block.data) for g in coupling.scopes]
        coupling.soft_threshold(block_thresholds, conditioners=conditioners)
        after = [torch.linalg.vector_norm(g.block.data) for g in coupling.scopes]

        # Each parameter norm should not increase
        for b, a in zip(before, after):
            assert a <= b + 1e-6
        # With random nonzero data and nonzero threshold, at least one norm decreased
        assert any(a < b - 1e-6 for b, a in zip(before, after))

    def test_soft_threshold_does_not_increase_block_norms(self, coupling):
        block_thresholds = torch.full(coupling.grid_shape, 0.1)
        conditioners = {g.block: torch.ones_like(g.block.data) for g in coupling.scopes}

        before = [g.block.norms(None).clone() for g in coupling.scopes]
        coupling.soft_threshold(block_thresholds, conditioners=conditioners)
        after = [g.block.norms(None).clone() for g in coupling.scopes]

        for b, a in zip(before, after):
            assert torch.all(a <= b + 1e-6)
        # With random nonzero data, at least some block norms should strictly decrease
        assert any(torch.any(a < b - 1e-6) for b, a in zip(before, after))
