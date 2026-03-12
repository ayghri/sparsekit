import pytest
import torch
from torch.nn import Parameter
from sparsekit.block import BlockSpec
from sparsekit.block import BlockCoupling
from sparsekit.utils import ShapeMismatchError
import math


class TestSoftThresholdAdam:
    @pytest.fixture
    def spec_2x2(self):
        # 2x2 tensor, single block
        # Initialize with ones. Norm = 2.0
        W = torch.ones(2, 2)
        param = Parameter(W)
        return BlockSpec(param, shape=(2, 2))

    def test_adam_equivalence_to_euclidean(self, spec_2x2):
        """
        Test that Adam soft thresholding reduces to Euclidean when conditioners (H) are 1.
        We choose parameters such that denom = norm - threshold > 1 to avoid the hard cutoff in the code.

        Setup:
        W = ones(2,2), L2 norm = 2.0.
        H = ones(2,2).
        Threshold = 0.5.

        Logic:
        denom = 2.0 - 0.5 = 1.5 (> 0.0, so it survives).
        Equation: mu / (1 + mu) * ||W|| = threshold
                  mu / (1 + mu) * 2 = 0.5
                  2 * mu = 0.5 * (1 + mu)
                  1.5 * mu = 0.5
                  mu = 1/3

        Scaling factor: H / (H + mu) = 1 / (1 + 1/3) = 3/4 = 0.75.
        Expected result: 0.75 * W = 0.75.
        """
        conditioners = torch.ones_like(spec_2x2.data)
        thresholds = torch.tensor([0.5]).reshape(spec_2x2.grid_shape)

        spec_2x2._soft_threshold_diag_cond(thresholds, conditioners)

        expected = torch.full((2, 2), 0.75)
        assert torch.allclose(spec_2x2.data, expected, atol=1e-5)

    def test_adam_varying_conditioner(self, spec_2x2):
        """
        Test with non-identity conditioners.

        Setup:
        W = ones(2,2).
        H = 2 * ones(2,2).
        Threshold = 1.0.

        Logic:
        Weighted vector Hv = 2 * W. Norm = 2 * 2 = 4.
        denom = 4 - 1 = 3 (> 1, survives).

        Equation: mu * || H/(H+mu) * W || = threshold
                  mu * || 2/(2+mu) * W || = 1
                  mu * (2/(2+mu)) * ||W|| = 1
                  mu * (2/(2+mu)) * 2 = 1
                  4 * mu = 2 + mu
                  3 * mu = 2
                  mu = 2/3

        Scaling factor: H / (H + mu) = 2 / (2 + 2/3) = 2 / (8/3) = 6/8 = 0.75.
        Expected result: 0.75 * W = 0.75.
        """
        conditioners = torch.full_like(spec_2x2.data, 2.0)
        thresholds = torch.tensor([1.0]).reshape(spec_2x2.grid_shape)

        spec_2x2._soft_threshold_diag_cond(thresholds, conditioners)

        expected = torch.full((2, 2), 0.75)
        assert torch.allclose(spec_2x2.data, expected, atol=1e-5)

    def test_adam_high_threshold_zeros_out(self, spec_2x2):
        """
        Test that a high threshold zeros out the block.
        Norm = 2. Threshold = 3.
        denom = 2 - 3 = -1.
        Should be zeroed.
        """
        conditioners = torch.ones_like(spec_2x2.data)
        thresholds = torch.tensor([3.0]).reshape(spec_2x2.grid_shape)

        spec_2x2._soft_threshold_diag_cond(thresholds, conditioners)

        assert torch.allclose(spec_2x2.data, torch.zeros(2, 2))

    def test_adam_cutoff_behavior(self, spec_2x2):
        """
        Test the specific cutoff behavior in the code: denom > 0
        denom = norm - threshold.

        Case 1: norm=2, threshold=0.9. denom=1.1 (>0). Should survive.
        Case 2: norm=2, threshold=3.0. denom=-1.0 (not >0). Should zero out.
        """
        # Case 1: Survive
        spec_survive = BlockSpec(
            Parameter(torch.ones(2, 2)), shape=(2, 2)
        )
        cond = torch.ones_like(spec_survive.data)
        thresh_survive = torch.tensor([0.9]).reshape(
            spec_survive.grid_shape
        )
        spec_survive._soft_threshold_diag_cond(thresh_survive, cond)
        assert not torch.allclose(spec_survive.data, torch.zeros(2, 2))

        # Case 2: Zero out
        spec_die = BlockSpec(Parameter(torch.ones(2, 2)), shape=(2, 2))
        thresh_die = torch.tensor([3.0]).reshape(spec_die.grid_shape)
        spec_die._soft_threshold_diag_cond(thresh_die, cond)
        assert torch.allclose(spec_die.data, torch.zeros(2, 2))

    def test_adam_shapes_mismatch(self, spec_2x2):
        """Test that shape mismatches raise assertions."""
        conditioners = torch.ones((3, 3))  # Wrong shape
        thresholds = torch.zeros(spec_2x2.grid_shape)

        with pytest.raises(ShapeMismatchError):
            spec_2x2._soft_threshold_diag_cond(thresholds, conditioners)

    def test_adam_multi_block_mixed(self):
        """
        Test a tensor with multiple blocks where some survive and some don't.
        4x4 tensor, 2x2 blocks.
        """
        param = Parameter(torch.ones(4, 4))
        spec = BlockSpec(param, shape=(2, 2))

        # Conditioners: All 1s
        H = torch.ones(4, 4)

        # Thresholds:
        # Top-Left: 0.5 (Norm=2, denom=1.5 > 1 -> Survive)
        # Bottom-Right: 3.0 (Norm=2, denom=-1 -> Die)
        thresholds = torch.tensor([[0.5, 3.0], [3.0, 3.0]])

        spec._soft_threshold_diag_cond(thresholds, H)

        # Top-Left should be non-zero (specifically 0.75 as calculated before)
        assert torch.allclose(
            spec.data[0:2, 0:2], torch.full((2, 2), 0.75), atol=1e-5
        )

        # Bottom-Right should be zero
        assert torch.allclose(spec.data[2:4, 2:4], torch.zeros(2, 2))

    def test_adam_one_block_var(self):
        """
        Test a tensor with one blocks where conditioner is not uniform
        """
        # we need 1.0 = u^2 (sum_i (h_i v_i/(h_i+u))^2 )
        # since ||hv|| = (5.3125)^.5 = 2.3048 > 1.0, a solution exists
        # we get mu = 1.16917
        # and w = (h / (h+u)) * v = 1/mu

        h = torch.tensor([[0.25, 0.5], [1.0, 2.0]])
        v = Parameter(torch.ones(2, 2))
        spec = BlockSpec(v, shape=(2, 2))
        thresholds = torch.tensor([1.0]).reshape(spec.grid_shape)

        mu = 1.1691705341
        expected = v * h / (h + mu)

        spec._soft_threshold_diag_cond(thresholds, h, max_iter=20)

        assert torch.allclose(spec.data, expected)

    def test_adam_one_block_solvable(self):
        h = torch.tensor([0.49671415, 0.1382643, 0.64768854, 1.52302986])
        v = Parameter(torch.ones(4))

        spec = BlockSpec(v, shape=(4,))
        thresholds = torch.tensor([1.0])

        mu = 1.6383774184
        expected = v * h / (h + mu)

        spec._soft_threshold_diag_cond(thresholds, h, max_iter=20)

        assert torch.allclose(spec.data, expected)

    def test_adam_one_block_unsolvable(self):
        h = torch.tensor([0.5, 0.5, 0.5, 0.5])
        v = Parameter(torch.tensor([0.9, 0.9, 0.9, 0.9]))

        spec = BlockSpec(v, shape=(4,))
        thresholds = torch.tensor([1.0])

        expected = torch.zeros_like(v)

        spec._soft_threshold_diag_cond(thresholds, h, max_iter=20)

        assert torch.allclose(spec.data, expected)

    def test_adam_multi_blocks(self):
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

        spec = BlockSpec(v, shape=(1, 2, 2))
        thresholds = torch.tensor([0.5, 1.0, 1.0, 1.0]).reshape(spec.grid_shape)
        expected = torch.stack(
            [
                torch.full((2, 2), 0.75),
                v.data[1] * h[1] / (h[1] + 1.1691705341),
                v.data[2] * h[2] / (h[2] + 1.6383774184),
                torch.zeros((2, 2)),
            ]
        )
        spec._soft_threshold_diag_cond(thresholds, h, max_iter=50)

        assert torch.allclose(spec.data, expected)


class TestBlockSpecBasics:
    def test_block_grid_shape_and_num_blocks_2x2(self):
        p = Parameter(torch.zeros(4, 4))
        spec = BlockSpec(p, shape=(2, 2))

        # 4x4 with 2x2 blocks -> 2x2 grid
        assert spec.grid_shape == (2, 2)
        assert spec.num_blocks == 4
        assert spec.numel() == 4

    def test_block_grid_shape_single_block(self):
        p = Parameter(torch.zeros(4, 4))
        spec = BlockSpec(p, shape=(4, 4))

        # Single block -> grid (1,1)
        assert spec.grid_shape == (1, 1)
        assert spec.num_blocks == 1
        assert spec.numel() == 16

    def test_invalid_block_shape_dimension_mismatch(self):
        p = Parameter(torch.zeros(4, 4))
        with pytest.raises(ValueError):
            BlockSpec(p, shape=(2,))  # ndim mismatch

    def test_invalid_block_shape_not_divisible(self):
        p = Parameter(torch.zeros(5, 4))
        with pytest.raises(ValueError):
            BlockSpec(p, shape=(2, 2))

    def test_block_view_and_block_to_element_roundtrip(self):
        p = Parameter(torch.arange(16.0).view(4, 4))
        spec = BlockSpec(p, shape=(2, 2))

        # block_view without merge: (4,4) -> (2,2,2,2)
        view = spec.block_view(spec.data, reorder=False)
        assert view.shape == (2, 2, 2, 2)

        # block_norm should match manual computation
        norms = spec.norms(spec.data)
        assert norms.shape == spec.grid_shape

        # Broadcast a simple per-block multiplier and ensure shape
        block_vals = torch.ones(spec.grid_shape)
        full = spec.broadcast_block_to_element(block_vals)
        assert full.shape == spec.view.shape

    def test_apply_mask_and_multiplier(self):
        p = Parameter(torch.ones(4, 4))
        spec = BlockSpec(p, shape=(2, 2))

        # Mask out one block (top-left)
        mask = torch.zeros(spec.grid_shape, dtype=torch.bool)
        mask[0, 0] = True
        spec.apply_mask(mask)

        # Top-left should be zero, others unchanged
        assert torch.allclose(spec.data[0:2, 0:2], torch.zeros(2, 2))
        assert torch.allclose(spec.data[0:2, 2:4], torch.ones(2, 2))

        # Now apply a multiplier on the remaining blocks
        mult = torch.ones(spec.grid_shape)
        mult[0, 1] = 2.0
        spec.apply_multiplier(mult)

        # Top-right block should be scaled by 2
        assert torch.allclose(spec.data[0:2, 2:4], torch.full((2, 2), 2.0))


class TestSparseNodeSoftThreshold:
    def test_soft_threshold_delegates_to_euclidean_when_no_conditioners(self):
        # Small tensor with two blocks
        p = Parameter(torch.ones(4, 4))
        spec = BlockSpec(p, shape=(2, 2))

        # Thresholds chosen so that all blocks survive partially
        thresholds = torch.full(spec.grid_shape, 0.5)
        before = spec.data.clone()

        spec.soft_threshold(thresholds, conditioners=None)

        # Euclidean soft-threshold should shrink but not zero everything
        assert not torch.allclose(spec.data, before)
        assert torch.any(spec.data != 0.0)


class TestBlockCoupling:
    @pytest.fixture
    def coupling_simple(self):
        # Two 1x1 tensors coupled together
        # W1 = [3.0], W2 = [4.0] -> Group Norm = 5.0
        p1 = Parameter(torch.tensor([[3.0]]))
        p2 = Parameter(torch.tensor([[4.0]]))
        s1 = BlockSpec(p1, shape=(1, 1))
        s2 = BlockSpec(p2, shape=(1, 1))

        coupling = BlockCoupling([s1, s2], orders=[])
        return coupling, s1, s2

    def test_coupling_adam_equivalence_simple(self, coupling_simple):
        """
        Test simple case with identity conditioners.
        W1=3, W2=4. Norm=5.
        Threshold=2.5.
        Equation: mu/(1+mu) * 5 = 2.5 => mu=1.
        Scaling = 1/(1+1) = 0.5.
        Expected: W1=1.5, W2=2.0.
        """
        coupling, s1, s2 = coupling_simple

        conditioners = {
            s1: torch.ones_like(s1.data),
            s2: torch.ones_like(s2.data),
        }
        thresholds = torch.tensor([2.5]).reshape(coupling.grid_shape)

        coupling._soft_threshold_diag_cond(thresholds, conditioners)

        assert torch.allclose(s1.data, torch.tensor([[1.5]]))
        assert torch.allclose(s2.data, torch.tensor([[2.0]]))

    def test_coupling_adam_varying_conditioner(self, coupling_simple):
        """
        W1=3, W2=4.
        H1=2, H2=2.
        Weighted: HW1=6, HW2=8. Norm=10.
        Threshold=5.0.

        Equation: mu * || H/(H+mu) * W || = threshold
        Since H is constant 2:
        mu * (2/(2+mu)) * ||W|| = 5
        mu * (2/(2+mu)) * 5 = 5
        2*mu / (2+mu) = 1
        2*mu = 2 + mu => mu = 2.

        Scaling = H/(H+mu) = 2/(2+2) = 0.5.
        Expected: W1=1.5, W2=2.0.
        """
        coupling, s1, s2 = coupling_simple

        conditioners = {
            s1: torch.full_like(s1.data, 2.0),
            s2: torch.full_like(s2.data, 2.0),
        }
        thresholds = torch.tensor([5.0]).reshape(coupling.grid_shape)

        coupling._soft_threshold_diag_cond(thresholds, conditioners)

        assert torch.allclose(s1.data, torch.tensor([[1.5]]))
        assert torch.allclose(s2.data, torch.tensor([[2.0]]))

    def test_coupling_adam_zeros_out(self, coupling_simple):
        """
        W1=3, W2=4. Norm=5.
        Threshold=6.0.
        denom = 5 - 6 = -1.
        Should zero out.
        """
        coupling, s1, s2 = coupling_simple
        conditioners = {
            s1: torch.ones_like(s1.data),
            s2: torch.ones_like(s2.data),
        }
        thresholds = torch.tensor([6.0]).reshape(coupling.grid_shape)

        coupling._soft_threshold_diag_cond(thresholds, conditioners)

        assert torch.allclose(s1.data, torch.zeros_like(s1.data))
        assert torch.allclose(s2.data, torch.zeros_like(s2.data))

    def test_coupling_mixed_blocks(self):
        """
        Two blocks.
        Block 1: W1=[3], W2=[4] (Norm 5). Threshold 2.5 -> Survives (scaled 0.5).
        Block 2: W1=[3], W2=[4] (Norm 5). Threshold 6.0 -> Dies.

        Tensors will be shape (2,1).
        """
        p1 = Parameter(torch.tensor([[3.0], [3.0]]))
        p2 = Parameter(torch.tensor([[4.0], [4.0]]))
        s1 = BlockSpec(p1, shape=(1, 1))
        s2 = BlockSpec(p2, shape=(1, 1))

        coupling = BlockCoupling([s1, s2], orders=[])

        conditioners = {
            s1: torch.ones_like(s1.data),
            s2: torch.ones_like(s2.data),
        }
        # Thresholds shape matches coupling.grid_shape
        thresholds = torch.tensor([2.5, 6.0]).reshape(coupling.grid_shape)

        coupling._soft_threshold_diag_cond(thresholds, conditioners)

        # First block scaled by 0.5
        assert torch.allclose(s1.data[0], torch.tensor([1.5]))
        assert torch.allclose(s2.data[0], torch.tensor([2.0]))

        # Second block zeroed
        assert torch.allclose(s1.data[1], torch.tensor([0.0]))
        assert torch.allclose(s2.data[1], torch.tensor([0.0]))

    def test_coupling_different_shapes(self):
        """
        Test coupling tensors of different shapes but compatible block grids.
        s1: (2,2), block (2,2) -> grid (1,1)
        s2: (1,4), block (1,4) -> grid (1,1)
        """
        p1 = Parameter(torch.ones(2, 2))  # Norm 2
        p2 = Parameter(torch.ones(1, 4))  # Norm 2
        # Combined norm = sqrt(4 + 4) = sqrt(8) approx 2.828

        s1 = BlockSpec(p1, shape=(2, 2))
        s2 = BlockSpec(p2, shape=(1, 4))

        coupling = BlockCoupling([s1, s2], orders=[])

        # Threshold 1.0.
        # mu/(1+mu) * sqrt(8) = 1.0
        # mu = 1/(sqrt(8)-1) approx 0.5469
        # scale = 1/(1+mu) = (sqrt(8)-1)/sqrt(8) = 1 - 1/sqrt(8) approx 0.6464

        thresholds = torch.tensor([[1.0]])
        conditioners = {
            s1: torch.ones_like(s1.data),
            s2: torch.ones_like(s2.data),
        }

        coupling._soft_threshold_diag_cond(thresholds, conditioners)

        scale = 1.0 - 1.0 / math.sqrt(8)
        assert torch.allclose(s1.data, torch.full((2, 2), scale), atol=1e-4)
        assert torch.allclose(s2.data, torch.full((1, 4), scale), atol=1e-4)

    def test_coupling_mismatched_block_grids_raises(self):
        """Specs with incompatible block grids should fail during construction."""
        p1 = Parameter(torch.ones(4, 4))
        p2 = Parameter(torch.ones(4, 4))

        # Make specs whose block_grid_shapes, once permuted, cannot match
        s1 = BlockSpec(p1, shape=(2, 2))  # block_grid_shape (2,2)
        s2 = BlockSpec(
            p2, shape=(2, 2)
        )  # same grid but we'll use bad order

        # First spec uses identity order, second uses invalid permutation length
        with pytest.raises(ValueError):
            BlockCoupling([s1, s2], orders=[(0, 1), (0,)])

    def test_coupling_apply_mask_and_multiplier(self, coupling_simple):
        coupling, s1, s2 = coupling_simple

        # Mask out first block
        mask = torch.tensor([True]).reshape(coupling.grid_shape)
        coupling.apply_mask(mask)
        assert torch.allclose(s1.data, torch.zeros_like(s1.data))
        assert torch.allclose(s2.data, torch.zeros_like(s2.data))

        # Reset and test multiplier
        s1.data.fill_(1.0)
        s2.data.fill_(2.0)
        mult = torch.tensor([2.0]).reshape(coupling.grid_shape)
        coupling.apply_multiplier(mult)
        assert torch.allclose(s1.data, torch.full_like(s1.data, 2.0))
        assert torch.allclose(s2.data, torch.full_like(s2.data, 4.0))

    def test_coupling_mixed_blocks_3d(self):
        """
        Two blocks.
        Block 1: W1=[3], W2=[4] (Norm 5). Threshold 2.5 -> Survives (scaled 0.5).
        Block 2: W1=[3], W2=[4] (Norm 5). Threshold 6.0 -> Dies.

        Tensors will be shape (2,1).
        """
        p1 = Parameter(torch.ones(4, 5, 8))
        s1 = BlockSpec(p1, shape=(2, 1, 2))

        p2 = Parameter(torch.ones(8, 4, 5) / 2.0)
        s2 = BlockSpec(p2, shape=(2, 2, 1))

        coupling = BlockCoupling([s1, s2], orders=[(0, 1, 2), (1, 2, 0)])

        conditioners = {
            s1: torch.ones_like(s1.data),
            s2: torch.ones_like(s2.data),
        }
        # Thresholds shape (2,1)
        thresholds = torch.ones((2, 5, 4)) * 1.0
        print(coupling._reverse_orders)

        coupling.soft_threshold(thresholds, conditioners)

        # First block scaled by 0.5
        assert torch.allclose(p1.data - 0.5528, torch.zeros_like(p1), atol=1e-4)

        # Second block zeroed
        assert torch.allclose(
            p2.data - 0.27645, torch.zeros_like(p2), atol=1e-4
        )
