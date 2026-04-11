Abstract Base Classes
=====================

These abstract classes define the interface shared by all block and scope
types.  You do not instantiate them directly — use :class:`~sparsekit.block.BlockSpec`,
:class:`~sparsekit.block.BlockCoupling`, :class:`~sparsekit.scope.ScopeSpec`,
or :class:`~sparsekit.scope.ScopeCoupling` instead.

SparseBlock
-----------

.. autoclass:: sparsekit.block.SparseBlock
   :members:
   :show-inheritance:

SparseScope
-----------

.. autoclass:: sparsekit.scope.SparseScope
   :members:
   :show-inheritance:
