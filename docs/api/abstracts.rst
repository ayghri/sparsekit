Abstract Base Classes
=====================

These abstract classes define the interface shared by all group and scope
types.  You do not instantiate them directly — use :class:`~sparsekit.block.BlockSpec`,
:class:`~sparsekit.block.BlockCoupling`, :class:`~sparsekit.scope.ScopeSpec`,
or :class:`~sparsekit.scope.ScopeCoupling` instead.

SparseNode
----------

.. autoclass:: sparsekit.block.SparseNode
   :members:
   :show-inheritance:

SparseScope
-----------

.. autoclass:: sparsekit.scope.SparseScope
   :members:
   :show-inheritance:
