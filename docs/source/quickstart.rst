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
   from sparsekit import BlockSpec, ScopeSpec, StructuredOBS

   W = torch.nn.Parameter(torch.randn(256, 1024, device="cuda"))
   X = torch.randn(4096, 1024, device="cuda")

   # 1. Build hierarchy
   block = BlockSpec(W, shape=(1, 1))              # scalar blocks
   scope = ScopeSpec(block, shape=(1, 4))      # scopes of 4 blocks

   # 2. Hessian and its inverse
   hessian = (X.T @ X) / X.shape[0]
   inv_h = StructuredOBS.compute_inverse(hessian, damp=1e-4)

   # 3. Prune (keep 2 of 4 blocks per scope)
   obs = StructuredOBS(scope, hessian, inv_h=inv_h)
   obs.prune(num_nz=2, compensate="local")         # fast, within-scope
   # obs.prune(num_nz=2, compensate="interleaved", n_splits=64)  # best quality

Magnitude Pruning (no Hessian)
------------------------------

.. code-block:: python

   from sparsekit import BlockSpec, ScopeSpec

   block = BlockSpec(W, shape=(1, 1))
   scope = ScopeSpec(block, shape=(1, 4))
   scope.hard_threshold(num_nz=2)   # keeps 2 largest-norm blocks per scope

Coupled 2:4 (Two Parameters)
-----------------------------

Prune two weight matrices jointly so their sparsity masks are coupled:

.. code-block:: python

   from sparsekit import BlockSpec, BlockCoupling, ScopeSpec, ScopeCoupling

   U = torch.nn.Parameter(torch.randn(4, 8, 2, 2, device="cuda"))
   V = torch.nn.Parameter(torch.randn(8, 16, 2, 2, device="cuda"))

   block_u = BlockSpec(U, shape=(2, 2, 2, 2), name="U")
   block_v = BlockSpec(V, shape=(2, 2, 2, 2), name="V")

   scope_u = ScopeSpec(block_u, shape=(1, 1), name="pU")
   scope_v = ScopeSpec(block_v, shape=(1, 4), name="pV")

   coupled = ScopeCoupling(
       [scope_u, scope_v],
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
       .add_scope("U", scope_shape=(1, 1), name="pU")
       .add_scope("V", scope_shape=(1, 4), name="pV")
       .couple_scopes(["pU", "pV"], orders=[(0, 1), (1, 0)], name="UV")
   )
   coupling = builder.get_scope("UV")

Sparsity Patterns
-----------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 35

   * - Pattern
     - block shape
     - scope shape
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
     - Pair columns 8 apart in 16-col segments
   * - Block-16 coupled
     - ``(1, 1, 16)``
     - ``(1, 2, 1)``
     - 16-col blocks, 8-row coupling
