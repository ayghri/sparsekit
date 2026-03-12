Concepts
========

This page explains the core abstractions and how they compose to
express arbitrary structured sparsity patterns.

The Hierarchy
-------------

Sparsekit uses a four-level hierarchy::

   nn.Parameter
     └── View             (strided view, write-through)
           └── BlockSpec   (block grid, single parameter)
                 └── GroupSpec   (groups of blocks, pruning unit)

Each level adds structure on top of the previous one:

1. **View** -- a strided ``as_strided`` view of a ``nn.Parameter``
2. **BlockSpec** -- divides the view into a grid of blocks
3. **GroupSpec** -- groups blocks into competition units for pruning
4. **StructuredOBS** -- performs OBS pruning through the GroupSpec

For multi-parameter sparsity, there are coupling variants:

- **BlockCoupling** -- couples multiple ``BlockSpec`` objects
- **GroupCoupling** -- couples multiple ``GroupSpec`` objects

View: Strided Write-Through Views
----------------------------------

:class:`~sparsekit.view.View` wraps an ``nn.Parameter`` with an
arbitrary ``(shape, stride)`` view using ``torch.as_strided``.  The key
property is **write-through**: pruning and masking operations modify the
underlying parameter storage directly, without copying the weight tensor.
Intermediate computations (norms, thresholds, mask broadcasting) may
allocate temporaries, but the weights themselves are never duplicated.

.. code-block:: python

   from sparsekit import View

   param = torch.nn.Parameter(torch.randn(2560, 9728))
   # View as (M, K/16, 8, 2) with stride (K, 16, 1, 8)
   view = View(param, shape=(2560, 608, 8, 2), stride=(9728, 16, 1, 8))

This is essential for coupled sparsity patterns where elements that are
far apart in memory must share a pruning decision.

BlockSpec: Block Grids
-----------------------

:class:`~sparsekit.block.BlockSpec` treats a tensor (or View) as a
grid of blocks.  Each block is a small sub-tensor defined by
``shape``.

.. code-block:: python

   from sparsekit import BlockSpec

   param = torch.nn.Parameter(torch.randn(8, 16))
   block = BlockSpec(param, shape=(2, 4))
   # grid_shape = (4, 4), block_numel = 8

Key operations:

- ``norms(values)`` -- L2 norm per block
- ``hard_threshold(thresholds)`` -- zero blocks below threshold
- ``soft_threshold(thresholds)`` -- proximal L1 operator
- ``get_masks(block_masks)`` -- convert block mask to element mask
- ``apply_multiplier(multiplier)`` -- scale each block

All threshold/mask operations write through to the parameter.

GroupSpec: Groups of Blocks
----------------------------

:class:`~sparsekit.group.GroupSpec` partitions the block grid into groups.
Within each group, blocks compete based on their norms: the top-``num_nz``
blocks survive; the rest are pruned.

.. code-block:: python

   from sparsekit import BlockSpec, GroupSpec

   block = BlockSpec(param, shape=(1, 1))   # scalar blocks
   group = GroupSpec(block, shape=(1, 4))    # 4 blocks per group
   group.hard_threshold(num_nz=2)           # keep 2 of 4

The ``shape`` specifies how many blocks along each dimension form one
group.  Use ``-1`` to span the entire dimension.

Key operations:

- ``block_to_group(b)`` -- reshape block tensor to group layout
- ``group_to_block(g)`` -- broadcast group values back to block grid
- ``block_norms(values)`` -- block norms in group layout
- ``kth_largest(values, num_nz)`` -- per-group pruning thresholds
- ``hard_threshold(num_nz=...)`` -- prune in-place
- ``get_masks(num_nz)`` -- return element-level masks without pruning

BlockCoupling and GroupCoupling
--------------------------------

When sparsity must be shared across parameters (e.g., coupled 2:4 where
column pairs 8 apart must have identical masks), use the coupling classes.

:class:`~sparsekit.block.BlockCoupling` merges multiple BlockSpec objects
into one virtual block grid.  The ``orders`` parameter specifies dimension
permutations to align their grids.

:class:`~sparsekit.group.GroupCoupling` does the same at the group level:
it concatenates block norms from all child GroupSpec objects along the last
dimension, then applies a single threshold across all of them.

.. code-block:: python

   from sparsekit import BlockSpec, GroupSpec, GroupCoupling

   block_a = BlockSpec(param_a, shape=(2, 2), name="A")
   block_b = BlockSpec(param_b, shape=(2, 2), name="B")

   group_a = GroupSpec(block_a, shape=(1, 1), name="gA")
   group_b = GroupSpec(block_b, shape=(1, 4), name="gB")

   coupled = GroupCoupling(
       [group_a, group_b],
       orders=[(0, 1), (1, 0)],   # align group grids
   )
   coupled.hard_threshold(num_nz=2)

StructuredOBS: Optimal Brain Surgeon
-------------------------------------

:class:`~sparsekit.pruners.obs.StructuredOBS` implements the OBS pruning
algorithm using the GroupSpec abstraction.  It uses the inverse Hessian
``C = (H + damp*I)^{-1}`` to:

1. **Select** which blocks to prune (minimize OBS cost)
2. **Compensate** remaining weights to reduce the pruning error

Compensation modes:

- ``"local"`` -- compensate within each group only (fast, independent)
- ``"full"`` -- sequential compensation to all K columns via ``C[P, :]``
- ``"split"`` -- like ``"full"`` but recomputes C between column splits
- ``"interleaved"`` -- re-selects masks AND compensates at each split
  (highest quality, nearly matches SparseGPT)

.. code-block:: python

   from sparsekit import BlockSpec, GroupSpec, StructuredOBS

   block = BlockSpec(W, shape=(1, 1))
   group = GroupSpec(block, shape=(1, 4))

   H = (X.T @ X) / X.shape[0]
   C = StructuredOBS.compute_inverse(H, damp=1e-4)

   obs = StructuredOBS(group, H, C=C)
   obs.prune(num_nz=2, compensate="interleaved", n_splits=64)
