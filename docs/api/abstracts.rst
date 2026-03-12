Abstract Base Classes
=====================

These abstract classes define the interface shared by all block and group
types.  You do not instantiate them directly — use :class:`~sparsekit.block.BlockSpec`,
:class:`~sparsekit.block.BlockCoupling`, :class:`~sparsekit.group.GroupSpec`,
or :class:`~sparsekit.group.GroupCoupling` instead.

SparseNode
----------

.. autoclass:: sparsekit.block.SparseNode
   :members:
   :show-inheritance:

SparseGroup
-----------

.. autoclass:: sparsekit.group.SparseGroup
   :members:
   :show-inheritance:
