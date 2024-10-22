"""This subpackage implements the simulation of trees
"""

from ._simulate import generate_trees
from ._fmat import sample_spike_and_slab, generate_fmat

__all__ = ["generate_trees", "sample_spike_and_slab", "generate_fmat"]
