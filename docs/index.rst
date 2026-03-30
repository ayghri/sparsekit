Structured Sparsity Specification Kit
=====================================

**This is a PyTorch implementation of the** :math:`S^3` **framework.**

Sparsekit provides a composable hierarchy for specifying and enforcing
structured sparsity patterns -- from 2:4 and 4:8 to coupled multi-parameter
patterns.  Pruning and masking write through directly to the
original ``nn.Parameter`` storage via ``torch.as_strided`` --
no weight copies are made.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   quickstart
   concepts
   results

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/views
   api/blocks
   api/scope
   api/builder
   api/obs
   api/linalg
   api/abstracts
