"""This subpackage contains utility functions implemented in JAX.
All functions are JIT-compiled.
"""

from ._ilp import ilp_jax
from ._polylog import polylog_jax

__all__ = ["ilp_jax", "polylog_jax"]
