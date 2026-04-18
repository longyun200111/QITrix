"""
Core linear-algebra utilities for finite-dimensional quantum information.

The public API exposes tensor products, subsystem permutations, partial trace,
partial transpose, and basic Fock-state constructors.
"""

from ._version import __version__
from .core import Operator, Space, SpaceList, operator
from .ops import extend, fock, fock_dm, permute, ptrace, ptrans, tensor

__all__ = [
    "__version__",
    "Operator",
    "operator",
    "Space",
    "SpaceList",
    "extend",
    "permute",
    "ptrace",
    "ptrans",
    "fock",
    "fock_dm",
    "tensor",
]
