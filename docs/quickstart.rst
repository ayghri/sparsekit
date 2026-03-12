Quick Start
===========

Installation
------------

Or from source:

.. code-block:: bash
   cd sparsekit && pip install .

Basic 2:4 Pruning
-----------------

Keep 2 of every 4 contiguous columns (50 % sparse, hardware-friendly on
NVIDIA Ampere+):

.. code-block:: python

   import torch
   from sparsekit import BlockSpec, GroupSpec, StructuredOBS

   W = torch.nn.Parameter(torch.randn(256, 1024, device="cuda"))
   X = torch.randn(4096, 1024, device="cuda")

   # 1. Build hierarchy
   block = BlockSpec(W, shape=(1, 1))        # scalar blocks
   group = GroupSpec(block, shape=(1, 4))     # groups of 4 columns

   # 2. Hessian and its inverse
   H = (X.T @ X) / X.shape[0]
   C = StructuredOBS.compute_inverse(H, damp=1e-4)

   # 3. Prune (keep 2 of 4 blocks per group)
   obs = StructuredOBS(group, H, C=C)
   obs.prune(num_nz=2, compensate="local")         # fast, within-group
   # obs.prune(num_nz=2, compensate="interleaved", n_splits=64)  # best quality

Magnitude Pruning (no Hessian)
------------------------------

.. code-block:: python

   from sparsekit import BlockSpec, GroupSpec

   block = BlockSpec(W, shape=(1, 1))
   group = GroupSpec(block, shape=(1, 4))
   group.hard_threshold(num_nz=2)   # keeps 2 largest-norm blocks per group

Coupled 2:4 (Two Parameters)
-----------------------------

Prune two weight matrices jointly so their sparsity masks are coupled:

.. code-block:: python

   from sparsekit import BlockSpec, BlockCoupling, GroupSpec, GroupCoupling

   U = torch.nn.Parameter(torch.randn(4, 8, 2, 2, device="cuda"))
   V = torch.nn.Parameter(torch.randn(8, 16, 2, 2, device="cuda"))

   block_u = BlockSpec(U, shape=(2, 2, 2, 2), name="U")
   block_v = BlockSpec(V, shape=(2, 2, 2, 2), name="V")

   group_u = GroupSpec(block_u, shape=(1, 1), name="gU")
   group_v = GroupSpec(block_v, shape=(1, 4), name="gV")

   coupled = GroupCoupling(
       [group_u, group_v],
       orders=[(0, 1), (1, 0)],
   )
   coupled.hard_threshold(num_nz=2)

Using the Builder API
---------------------

.. code-block:: python

   from sparsekit.builder import SparsityBuilder

   builder = (
       SparsityBuilder()
       .add_block(U, (2, 2, 2, 2), name="U")
       .add_block(V, (2, 2, 2, 2), name="V")
       .add_group("U", group_shape=(1, 1), name="gU")
       .add_group("V", group_shape=(1, 4), name="gV")
       .couple_groups(["gU", "gV"], orders=[(0, 1), (1, 0)], name="UV")
   )
   coupling = builder.get_group("UV")

Sparsity Patterns
-----------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 35

   * - Pattern
     - block shape
     - group shape
     - Description
   * - 2:4
     - ``(1, 1)``
     - ``(1, 4)``
     - Keep 2 of 4 contiguous columns
   * - 4:8
     - ``(1, 2)``
     - ``(1, 4)``
     - Keep 2 of 4 column-pairs
   * - Coupled 2:4
     - Via ``View``
     - ``(1, 1, 4, 1)``
     - Pair columns 8 apart in 16-col groups
   * - Block-16 coupled
     - ``(1, 1, 16)``
     - ``(1, 2, 1)``
     - 16-col blocks, 8-row coupling
