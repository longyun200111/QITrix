"""
Core linear-algebra utilities for finite-dimensional quantum information.

The public API exposes tensor products, subsystem permutations, partial trace,
partial transpose, and basic Fock-state constructors.
"""

from ._version import __version__
from .extend import extend
from .permute import permute
from .ptrace import ptrace
from .ptrans import ptrans
from .state import fock, fock_dm
from .tensor import tensor

__all__ = [
    "__version__",
    "extend",
    "permute",
    "ptrace",
    "ptrans",
    "fock",
    "fock_dm",
    "tensor",
]
