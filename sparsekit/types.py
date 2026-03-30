# Copyright (c) 2026 - Ayoub Ghriss & Contributors
# Licensed under CC BY-NC 4.0
# (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
# Non-commercial use only; contact us for commercial licensing.
"""Type aliases and exception classes for sparsekit."""

from typing import Any, Tuple, Union, Mapping

from torch import Tensor

Values = Union[Tensor, Mapping[Any, Tensor], None]


# ── Exceptions ───────────────────────────────────────────────────────


class SparseKitError(Exception):
    """Base exception for sparsekit errors."""


class ShapeMismatchError(SparseKitError):
    """Raised when tensor shapes do not match."""

    def __init__(
        self,
        expected: Tuple[int, ...],
        got: Tuple[int, ...],
        context: str = "",
    ):
        msg = f"Shape mismatch: expected {expected}, got {got}"
        super().__init__(
            f"{context}: {msg}" if context else msg
        )


class CouplingError(SparseKitError):
    """Raised when coupling constraints are violated."""
