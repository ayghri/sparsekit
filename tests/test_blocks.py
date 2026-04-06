import pytest
import torch
from torch.nn import Parameter
from sparsekit.block import BlockSpec
from sparsekit.view import View
from sparsekit.block import BlockCoupling
from sparsekit.block import ShapeMismatchError
import math


class TestSoftThresholdAdam:
    @pytest.fixture
    def spec_2x2(self):
        # 2x2 tensor, single group
        # Initialize with ones. Norm = 2.0
        W = torch.ones(2, 2)
        param = Parameter(W)
        return BlockSpec(View.from_existing(param), shape=(2, 2))

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

        spec_2x2._soft_threshold_diag_cond(thresholds, conditioners,
                                           max_iter=20,
                                           atol=1e-8,
                                           eps=1e-8)

        expected = torch.full((2, 2), 0.75)
        assert torch.allclose(spec_2x2.data, expected)

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

        spec_2x2._soft_threshold_diag_cond(thresholds, conditioners,
                                           max_iter=20,
                                           atol=1e-8,
                                           eps=1e-8)

        expected = torch.full((2, 2), 0.75)
        assert torch.allclose(spec_2x2.data, expected)

    def test_adam_high_threshold_zeros_out(self, spec_2x2):
        """
        Test that a high threshold zeros out the group.
        Norm = 2. Threshold = 3.
        denom = 2 - 3 = -1.
        Should be zeroed.
        """
        conditioners = torch.ones_like(spec_2x2.data)
        thresholds = torch.tensor([3.0]).reshape(spec_2x2.grid_shape)

        spec_2x2._soft_threshold_diag_cond(thresholds, conditioners,
                                           max_iter=20,
                                           atol=1e-8,
                                           eps=1e-8)
        assert torch.allclose(spec_2x2.data, torch.zeros(2, 2))

    def test_adam_cutoff_behavior(self, spec_2x2):
        """
        Test the specific cutoff behavior in the code: denom > 0
        denom = norm - threshold.

        Case 1: norm=2, threshold=0.9. denom=1.1 (>0). Should survive.
        Case 2: norm=2, threshold=3.0. denom=-1.0 (not >0). Should zero out.
        """
        # Case 1: Survive
        spec_survive = BlockSpec(Parameter(torch.ones(2, 2)), shape=(2, 2))
        cond = torch.ones_like(spec_survive.data)
        thresh_survive = torch.tensor([0.9]).reshape(spec_survive.grid_shape)
        spec_survive._soft_threshold_diag_cond(thresh_survive, cond,
                                           max_iter=20,
                                           atol=1e-8,
                                           eps=1e-8)
        assert not torch.allclose(spec_survive.data, torch.zeros(2, 2))
        # norm=2, threshold=0.9, H=1: mu/(1+mu)*2=0.9 → mu=0.9/1.1 → scale=1.1/2=0.55
        mu = 0.9 / 1.1
        expected_scale = 1.0 / (1.0 + mu)
        assert torch.allclose(
            spec_survive.data, torch.full((2, 2), expected_scale)
        )

        # Case 2: Zero out
        spec_die = BlockSpec(Parameter(torch.ones(2, 2)), shape=(2, 2))
        thresh_die = torch.tensor([3.0]).reshape(spec_die.grid_shape)
        spec_die._soft_threshold_diag_cond(thresh_die, cond,
                                           max_iter=20,
                                           atol=1e-8,
                                           eps=1e-8)
        assert torch.allclose(spec_die.data, torch.zeros(2, 2))

    def test_adam_shapes_mismatch(self, spec_2x2):
        """Test that shape mismatches raise assertions."""
        conditioners = torch.ones((3, 3))  # Wrong shape
        thresholds = torch.zeros(spec_2x2.grid_shape)

        with pytest.raises(ShapeMismatchError):
            spec_2x2._soft_threshold_diag_cond(thresholds, conditioners,
                                           max_iter=20,
                                           atol=1e-8,
                                           eps=1e-8)

    def test_adam_multi_block_mixed(self):
        """
        Test a tensor with multiple groups where some survive and some don't.
        4x4 tensor, 2x2 groups.
        """
        param = Parameter(torch.ones(4, 4))
        spec = BlockSpec(param, shape=(2, 2))

        # Conditioners: All 1s
        H = torch.ones(4, 4)

        # Thresholds:
        # Top-Left: 0.5 (Norm=2, denom=1.5 > 1 -> Survive)
        # Bottom-Right: 3.0 (Norm=2, denom=-1 -> Die)
        thresholds = torch.tensor([[0.5, 3.0], [3.0, 3.0]])

        spec._soft_threshold_diag_cond(thresholds, H,
                                           max_iter=20,
                                           atol=1e-8,
                                           eps=1e-8)

        # Top-Left should be non-zero (specifically 0.75 as calculated before)
        assert torch.allclose(spec.data[0:2, 0:2], torch.full((2, 2), 0.75))

        # Bottom-Right should be zero
        assert torch.allclose(spec.data[2:4, 2:4], torch.zeros(2, 2))

    def test_adam_one_block_var(self):
        """
        Test a tensor with one groups where conditioner is not uniform
        """
        # we need 1.0 = u^2 (sum_i (h_i v_i/(h_i+u))^2 )
        # since ||hv|| = (5.3125)^.5 = 2.3048 > 1.0, a solution exists
        # we get mu = 1.16917
        # and w = (h / (h+u)) * v = h / (h + mu)  (since v=ones)

        h = torch.tensor([[0.25, 0.5], [1.0, 2.0]])
        v = Parameter(torch.ones(2, 2))
        spec = BlockSpec(v, shape=(2, 2))
        thresholds = torch.tensor([1.0]).reshape(spec.grid_shape)

        mu = 1.1691705341
        expected = v * h / (h + mu)

        spec._soft_threshold_diag_cond(thresholds, h,
                                           max_iter=20,
                                           atol=1e-8,
                                           eps=1e-8)

        assert torch.allclose(spec.data, expected)

    def test_adam_one_block_solvable(self):
        h = torch.tensor([0.49671415, 0.1382643, 0.64768854, 1.52302986])
        v = Parameter(torch.ones(4))

        spec = BlockSpec(v, shape=(4,))
        thresholds = torch.tensor([1.0])

        mu = 1.6383774184
        expected = v * h / (h + mu)

        spec._soft_threshold_diag_cond(thresholds, h, max_iter=20,
                                           atol=1e-8,
                                           eps=1e-8)

        assert torch.allclose(spec.data, expected)

    def test_adam_one_block_unsolvable(self):
        h = torch.tensor([0.5, 0.5, 0.5, 0.5])
        v = Parameter(torch.tensor([0.9, 0.9, 0.9, 0.9]))

        spec = BlockSpec(v, shape=(4,))
        thresholds = torch.tensor([1.0])

        expected = torch.zeros_like(v)

        spec._soft_threshold_diag_cond(thresholds, h,
                                           max_iter=20,
                                           atol=1e-8,
                                           eps=1e-8)

        assert torch.allclose(spec.data, expected)

    def test_adam_multi_scopes(self):
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
        spec._soft_threshold_diag_cond(thresholds, h,
                                       max_iter=20,
                                           atol=1e-8,
                                           eps=1e-8)

        assert torch.allclose(spec.data, expected)


class TestBlockSpecBasics:
    def test_block_grid_shape_and_num_blocks_2x2(self):
        p = Parameter(torch.zeros(4, 4))
        spec = BlockSpec(p, shape=(2, 2))

        # 4x4 with 2x2 groups -> 2x2 grid
        assert spec.grid_shape == (2, 2)
        assert spec.num_blocks == 4
        assert spec.block_numel == 4

    def test_block_grid_shape_single_block(self):
        p = Parameter(torch.zeros(4, 4))
        spec = BlockSpec(p, shape=(4, 4))

        # Single group -> grid (1,1)
        assert spec.grid_shape == (1, 1)
        assert spec.num_blocks == 1
        assert spec.block_numel == 16

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
        # Block (0,0)=[[0,1],[4,5]], (0,1)=[[2,3],[6,7]], (1,0)=[[8,9],[12,13]], (1,1)=[[10,11],[14,15]]
        norms = spec.norms(spec.data)
        assert norms.shape == spec.grid_shape
        expected_norms = torch.tensor(
            [
                [
                    float((0**2 + 1**2 + 4**2 + 5**2) ** 0.5),
                    float((2**2 + 3**2 + 6**2 + 7**2) ** 0.5),
                ],
                [
                    float((8**2 + 9**2 + 12**2 + 13**2) ** 0.5),
                    float((10**2 + 11**2 + 14**2 + 15**2) ** 0.5),
                ],
            ]
        )
        assert torch.allclose(norms, expected_norms)

        # Broadcast a simple per-group multiplier and ensure shape
        block_vals = torch.ones(spec.grid_shape)
        full = spec.broadcast_block_to_element(block_vals)
        assert full.shape == spec.view.shape
        assert torch.allclose(full, torch.ones_like(full))

    def test_apply_mask_and_multiplier(self):
        p = Parameter(torch.ones(4, 4))
        spec = BlockSpec(p, shape=(2, 2))

        # Mask out one group (top-left)
        mask = torch.zeros(spec.grid_shape, dtype=torch.bool)
        mask[0, 0] = True
        spec.apply_mask(mask)

        # Top-left should be zero, others unchanged
        assert torch.allclose(spec.data[0:2, 0:2], torch.zeros(2, 2))
        assert torch.allclose(spec.data[0:2, 2:4], torch.ones(2, 2))

        # Now apply a multiplier on the remaining groups
        mult = torch.ones(spec.grid_shape)
        mult[0, 1] = 2.0
        spec.apply_multiplier(mult)

        # Top-right group should be scaled by 2
        assert torch.allclose(spec.data[0:2, 2:4], torch.full((2, 2), 2.0))


class TestSparseNodeSoftThreshold:
    def test_soft_threshold_delegates_to_euclidean_when_no_conditioners(self):
        # Small tensor with two groups
        p = Parameter(torch.ones(4, 4))
        spec = BlockSpec(p, shape=(2, 2))

        # Thresholds chosen so that all groups survive partially
        thresholds = torch.full(spec.grid_shape, 0.5)
        before = spec.data.clone()

        spec.soft_threshold(thresholds, conditioners=None)

        # Euclidean soft-threshold should shrink but not zero everything
        assert not torch.allclose(spec.data, before)
        assert torch.any(spec.data != 0.0)
        # norm=2, threshold=0.5: Euclidean factor = 1 - 0.5/2 = 0.75
        assert torch.allclose(spec.data, torch.full((4, 4), 0.75))


class TestBlockCoupling:
    @pytest.fixture
    def coupling_simple(self):
        # Two 1x1 tensors coupled together
        # W1 = [3.0], W2 = [4.0] -> Block Norm = 5.0
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

    def test_coupling_mixed_scopes(self):
        """
        Two groups.
        Group 1: W1=[3], W2=[4] (Norm 5). Threshold 2.5 -> Survives (scaled 0.5).
        Group 2: W1=[3], W2=[4] (Norm 5). Threshold 6.0 -> Dies.

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

        # First group scaled by 0.5
        assert torch.allclose(s1.data[0], torch.tensor([1.5]))
        assert torch.allclose(s2.data[0], torch.tensor([2.0]))

        # Second group zeroed
        assert torch.allclose(s1.data[1], torch.tensor([0.0]))
        assert torch.allclose(s2.data[1], torch.tensor([0.0]))

    def test_coupling_different_shapes(self):
        """
        Test coupling tensors of different shapes but compatible group grids.
        s1: (2,2), group (2,2) -> grid (1,1)
        s2: (1,4), group (1,4) -> grid (1,1)
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
        assert torch.allclose(s1.data, torch.full((2, 2), scale))
        assert torch.allclose(s2.data, torch.full((1, 4), scale))

    def test_coupling_mismatched_block_grids_raises(self):
        """Specs with incompatible group grids should fail during construction."""
        p1 = Parameter(torch.ones(4, 4))
        p2 = Parameter(torch.ones(4, 4))

        # Make specs whose block_grid_shapes, once permuted, cannot match
        s1 = BlockSpec(p1, shape=(2, 2))  # block_grid_shape (2,2)
        s2 = BlockSpec(p2, shape=(2, 2))  # same grid but we'll use bad order

        # First spec uses identity order, second uses invalid permutation length
        with pytest.raises(ValueError):
            BlockCoupling([s1, s2], orders=[(0, 1), (0,)])

    def test_coupling_apply_mask_and_multiplier(self, coupling_simple):
        coupling, s1, s2 = coupling_simple

        # Mask out first group
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

    def test_coupling_mixed_scopes_3d(self):
        """
        Two 3D tensor specs coupled via reordered grids.
        p1: ones(4,5,8), block (2,1,2) -> grid (2,5,4)
        p2: ones(8,4,5)/2, block (2,2,1) -> grid (4,2,5); coupled via (1,2,0) permutation
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

        # Combined block: 4 elements from p1 (val=1) + 4 from p2 (val=0.5), H=1
        # ||Hw|| = sqrt(4*1^2 + 4*0.5^2) = sqrt(5); threshold=1
        # mu/(1+mu)*sqrt(5)=1 → scale = 1/(1+mu) = 1 - 1/sqrt(5)
        scale = 1.0 - 1.0 / math.sqrt(5.0)
        assert torch.allclose(p1.data, torch.full_like(p1.data, scale))
        assert torch.allclose(p2.data, torch.full_like(p2.data, 0.5 * scale))


class TestBlockSoftThresholdNotebook:
    """Tests derived from playground/notebooks/test_soft.ipynb.

    The notebook validates three operating modes of BlockSpec.soft_threshold:
    1. Scalar mode (block_shape=(1,1)): reduces to element-wise L1 proximal.
    2. Block L2 mode (block_shape=(1,2), no conditioner): group proximal.
    3. Conditioned mode (block_shape=(1,2), diagonal H): weighted proximal.
    """

    def test_scalar_block_matches_elementwise_prox(self):
        # block_shape=(1,1): each element is its own block.
        # Euclidean soft-threshold on a size-1 block = scalar L1 proximal:
        #   sign(x) * max(|x| - λ, 0)
        x = torch.tensor(
            [
                [1.5410, -0.2934, -2.1788, 0.5684],
                [-1.0845, -1.3986, 0.4033, 0.8380],
                [-0.7193, -0.4033, -0.5966, 0.1820],
            ]
        )
        p = Parameter(x.clone())
        spec = BlockSpec(p, shape=(1, 1))

        lam = 0.1
        thresholds = torch.full(spec.grid_shape, lam)
        spec.soft_threshold(thresholds)

        expected = torch.sign(x) * torch.clamp(x.abs() - lam, min=0)
        assert torch.allclose(spec.data, expected, atol=1e-6)

    def test_block_l2_soft_threshold_no_conditioner(self):
        # block_shape=(1,2): blocks of 2 columns.
        # For each 2-element block g: result = (1 - λ/||g||₂)₊ * g
        x = torch.tensor(
            [
                [1.5410, -0.2934, -2.1788, 0.5684],
                [-1.0845, -1.3986, 0.4033, 0.8380],
                [-0.7193, -0.4033, -0.5966, 0.1820],
            ]
        )
        p = Parameter(x.clone())
        spec = BlockSpec(p, shape=(1, 2))

        lam = 0.1
        thresholds = torch.full(spec.grid_shape, lam)
        spec.soft_threshold(thresholds)

        # Reference: group proximal applied per (1,2) block
        x_view = x.view(3, 2, 2)
        norms = x_view.norm(dim=-1, keepdim=True)
        expected = (1.0 - lam / norms).clamp(min=0.0) * x_view
        assert torch.allclose(spec.data, expected.view(3, 4), atol=1e-6)

    def test_block_l2_zeros_small_norm_blocks(self):
        # Blocks whose L2 norm < threshold are fully zeroed.
        # Blocks whose L2 norm > threshold are shrunk but not zeroed.
        x = torch.tensor([[3.0, 4.0, 0.05, 0.0]])  # norms: 5.0, 0.05
        p = Parameter(x.clone())
        spec = BlockSpec(p, shape=(1, 2))

        lam = 0.1
        thresholds = torch.full(spec.grid_shape, lam)
        spec.soft_threshold(thresholds)

        # Block 0: norm=5 > 0.1 → survives: scale=(1-0.1/5)=0.98
        assert torch.allclose(
            spec.data[0, :2], torch.tensor([3.0, 4.0]) * 0.98, atol=1e-6
        )
        # Block 1: norm=0.05 < 0.1 → zeroed
        assert torch.allclose(spec.data[0, 2:], torch.zeros(2), atol=1e-6)

    def test_conditioned_block_soft_threshold_surviving(self):
        # block_shape=(1,2) with uniform H=2, x=[3,4], λ=1.
        # Algorithm finds μ s.t. zeta(μ) = μ * ||H*x/(H+μ)||₂ = λ.
        # With uniform H=h: zeta(μ) = μ*h/(h+μ) * ||x||₂
        #   μ*2*5/(2+μ)=1  →  10μ=2+μ  →  μ=2/9
        # result = H/(H+μ)*x = 2/(2+2/9)*[3,4] = (9/10)*[3,4] = [2.7, 3.6]
        x = torch.tensor([[3.0, 4.0]])
        H = torch.full_like(x, 2.0)

        p = Parameter(x.clone())
        spec = BlockSpec(p, shape=(1, 2))
        thresholds = torch.tensor([[1.0]])

        spec.soft_threshold(thresholds, conditioners=H)

        expected = torch.tensor([[2.7, 3.6]])
        assert torch.allclose(spec.data, expected, atol=1e-4)

    def test_conditioned_block_soft_threshold_zeroed(self):
        # block_shape=(1,2) with H=2, x=[0.02, 0.04], λ=0.1.
        # ||H*x||₂ = ||[0.04, 0.08]||₂ = sqrt(0.008) ≈ 0.0894 < 0.1
        # denom = ||H*x|| - λ < 0 → block is zeroed.
        x = torch.tensor([[0.02, 0.04]])
        H = torch.full_like(x, 2.0)

        p = Parameter(x.clone())
        spec = BlockSpec(p, shape=(1, 2))
        thresholds = torch.tensor([[0.1]])

        spec.soft_threshold(thresholds, conditioners=H)

        assert torch.allclose(spec.data, torch.zeros_like(x))

    def test_conditioned_fixed_point(self):
        # Notebook verification: result = H/(H+μ)*x for scalar μ within each block.
        # Block 0: x=[3,4], H=[1,4], λ=1 → survives (||H*x||₂=sqrt(9+256)>>1)
        # Block 1: x=[0.1,0.1], H=[1,1], λ=1 → ||H*x||₂=sqrt(0.02)<1 → zeroed
        x = torch.tensor([[3.0, 4.0, 0.1, 0.1]])
        H = torch.tensor([[1.0, 4.0, 1.0, 1.0]])

        p = Parameter(x.clone())
        spec = BlockSpec(p, shape=(1, 2))
        thresholds = torch.tensor([[1.0, 1.0]])

        spec.soft_threshold(thresholds, conditioners=H)
        result = spec.data

        # Block 1 zeroed
        assert torch.allclose(result[0, 2:], torch.zeros(2))

        # Block 0 surviving: verify result = H/(H+μ)*x for a consistent scalar μ
        # Extract μ per element: μ_i = H_i*(x_i/w_i - 1)
        w = result[0, :2]
        h = H[0, :2]
        xb = x[0, :2]
        mu_per_elem = h * (xb / w - 1)
        assert torch.allclose(mu_per_elem[0:1], mu_per_elem[1:2], atol=1e-4), (
            f"μ inconsistent across block elements: {mu_per_elem.tolist()}"
        )
        # And the zeta condition: μ * ||w||₂ == λ
        mu = mu_per_elem[0].item()
        assert abs(mu * w.norm().item() - 1.0) < 1e-4
