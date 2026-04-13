Concepts
========

This page explains the core abstractions and how they compose to
express arbitrary structured sparsity patterns.

The Hierarchy
-------------

Sparsekit uses a three-level hierarchy::

   nn.Parameter
    └── View        (strided view of the , write-through)
    └── BlockSpec   (block grid, single parameter)
    └── ScopeSpec   (scopes of blocks, pruning decision unit)

Each level adds structure on top of the previous one:

1. **View** -- a strided ``as_strided`` view of a ``nn.Parameter``
2. **BlockSpec** -- divides the view into a grid of blocks
3. **ScopeSpec** -- organizes blocks into scopes (competition units for pruning)
4. **StructuredOBS** -- performs OBS pruning through the ScopeSpec

For multi-parameter sparsity, there are coupling variants:

- **BlockCoupling** -- couples multiple ``BlockSpec`` objects
- **ScopeCoupling** -- couples multiple ``ScopeSpec`` objects

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

ScopeSpec: Scopes of Blocks
------------------------------------

:class:`~sparsekit.scope.ScopeSpec` divides the block grid into
competition scopes.  Within each scope, blocks compete based on their
norms: the top-``nnz`` blocks survive; the rest are pruned.

.. code-block:: python

   from sparsekit import BlockSpec, ScopeSpec

   block = BlockSpec(param, shape=(1, 1))        # scalar blocks
   scope = ScopeSpec(block, shape=(1, 4))    # 4 blocks per scope
   scope.hard_threshold(nnz=2)                # keep 2 of 4

The ``shape`` specifies how many blocks along each dimension form one
scope.  Use ``-1`` to span the entire dimension.

Key operations:

- ``block_to_scope(t)`` -- reshape block tensor to scope layout
- ``scope_to_block(t)`` -- broadcast scope values back to block grid
- ``block_norms(values)`` -- block norms in scope layout
- ``kth_largest(values, nnz)`` -- per-scope pruning thresholds
- ``hard_threshold(nnz=...)`` -- prune in-place
- ``get_masks(nnz)`` -- return element-level masks without pruning

BlockCoupling and ScopeCoupling
------------------------------------

When sparsity must be shared across parameters (e.g., coupled 2:4 where
column pairs 8 apart must have identical masks), use the coupling classes.

:class:`~sparsekit.block.BlockCoupling` merges multiple BlockSpec objects
into one virtual block grid.  The ``orders`` parameter specifies dimension
permutations to align their grids.

:class:`~sparsekit.scope.ScopeCoupling` does the same at the
scope level: it concatenates block norms from all child ScopeSpec
objects along the last dimension, then applies a single threshold across
all of them.

.. code-block:: python

   from sparsekit import BlockSpec, ScopeSpec, ScopeCoupling

   block_a = BlockSpec(param_a, shape=(2, 2), name="A")
   block_b = BlockSpec(param_b, shape=(2, 2), name="B")

   scope_a = ScopeSpec(block_a, shape=(1, 1), name="pA")
   scope_b = ScopeSpec(block_b, shape=(1, 4), name="pB")

   coupled = ScopeCoupling(
       [scope_a, scope_b],
       orders=[(0, 1), (1, 0)],   # align scope grids
   )
   coupled.hard_threshold(nnz=2)

StructuredOBS: Optimal Brain Surgeon
-------------------------------------

:class:`~sparsekit.pruners.obs.StructuredOBS` implements the OBS pruning
algorithm using the ScopeSpec abstraction.  It uses the inverse Hessian
``C = (H + damp*I)^{-1}`` to:

1. **Select** which blocks to prune (minimize OBS cost)
2. **Compensate** remaining weights to reduce the pruning error

Compensation modes:

- ``"local"`` -- compensate within each scope only (fast, independent)
- ``"full"`` -- sequential compensation to all K columns via ``C[P, :]``
- ``"split"`` -- like ``"full"`` but recomputes C between column splits
- ``"interleaved"`` -- re-selects masks AND compensates at each split
  (highest quality, nearly matches SparseGPT)

.. code-block:: python

   from sparsekit import BlockSpec, ScopeSpec, StructuredOBS

   block = BlockSpec(W, shape=(1, 1))
   scope = ScopeSpec(block, shape=(1, 4))

   hessian = (X.T @ X) / X.shape[0]
   inv_h = StructuredOBS.compute_inverse(hessian, damp=1e-4)

   obs = StructuredOBS(scope, hessian, inv_h=inv_h)
   obs.prune(nnz=2, compensate="interleaved", n_splits=64)
